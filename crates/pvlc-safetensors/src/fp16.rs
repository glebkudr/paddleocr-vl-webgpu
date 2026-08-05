use std::error::Error;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

use serde::Serialize;
use serde_json::{Map, Value};
use tempfile::NamedTempFile;

use crate::{COPY_BUFFER_BYTES, ImportError, SafetensorsCatalog};
use pvlc_model_schema::TensorDtype;

pub const FP16_CHECKPOINT_CONVERSION_ID: &str = "bf16_to_ieee_f16_rne_v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fp16CheckpointErrorCode {
    Io,
    InvalidSource,
    UnsupportedSourceDtype,
    NonFiniteSource,
    OutOfRangeSource,
    SourceChanged,
    SourceDestinationAlias,
    OutputAlreadyExists,
}

#[derive(Debug)]
pub struct Fp16CheckpointError {
    code: Fp16CheckpointErrorCode,
    tensor_name: Option<String>,
    element_index: Option<u64>,
    message: String,
    source: Option<Box<dyn Error + Send + Sync>>,
}

impl Fp16CheckpointError {
    #[must_use]
    pub const fn code(&self) -> Fp16CheckpointErrorCode {
        self.code
    }

    #[must_use]
    pub fn tensor_name(&self) -> Option<&str> {
        self.tensor_name.as_deref()
    }

    #[must_use]
    pub const fn element_index(&self) -> Option<u64> {
        self.element_index
    }

    fn new(code: Fp16CheckpointErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            tensor_name: None,
            element_index: None,
            message: message.into(),
            source: None,
        }
    }

    fn tensor(
        code: Fp16CheckpointErrorCode,
        tensor_name: impl Into<String>,
        element_index: Option<u64>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code,
            tensor_name: Some(tensor_name.into()),
            element_index,
            message: message.into(),
            source: None,
        }
    }

    fn io(error: io::Error, message: impl Into<String>) -> Self {
        Self {
            code: Fp16CheckpointErrorCode::Io,
            tensor_name: None,
            element_index: None,
            message: message.into(),
            source: Some(Box::new(error)),
        }
    }

    fn tensor_io(error: io::Error, tensor_name: &str, message: impl Into<String>) -> Self {
        Self {
            code: Fp16CheckpointErrorCode::Io,
            tensor_name: Some(tensor_name.to_owned()),
            element_index: None,
            message: message.into(),
            source: Some(Box::new(error)),
        }
    }

    fn with_source(
        code: Fp16CheckpointErrorCode,
        message: impl Into<String>,
        source: impl Error + Send + Sync + 'static,
    ) -> Self {
        Self {
            code,
            tensor_name: None,
            element_index: None,
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }
}

impl fmt::Display for Fp16CheckpointError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "FP16 checkpoint conversion error {:?}: {}",
            self.code, self.message
        )?;
        if let Some(name) = &self.tensor_name {
            write!(formatter, " (tensor {name:?}")?;
            if let Some(index) = self.element_index {
                write!(formatter, ", element {index}")?;
            }
            write!(formatter, ")")?;
        }
        Ok(())
    }
}

impl Error for Fp16CheckpointError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn Error + 'static))
    }
}

impl From<ImportError> for Fp16CheckpointError {
    fn from(error: ImportError) -> Self {
        let tensor_name = error.tensor_name().map(str::to_owned);
        Self {
            code: Fp16CheckpointErrorCode::InvalidSource,
            tensor_name,
            element_index: None,
            message: error.to_string(),
            source: Some(Box::new(error)),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Fp16PayloadConversionReport {
    pub element_count: u64,
    pub max_buffer_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Fp16CheckpointConversionReport {
    pub conversion: &'static str,
    pub tensor_count: u64,
    pub element_count: u64,
    pub source_bytes: u64,
    pub output_bytes: u64,
    pub source_blake3: String,
    pub output_blake3: String,
    pub max_payload_buffer_bytes: usize,
}

/// Converts a finite BF16 bit pattern into IEEE-754 binary16 using
/// round-to-nearest, ties-to-even. Values that would become nonfinite return
/// `None`; BF16 NaN and infinity also return `None`.
#[must_use]
pub fn finite_bf16_to_f16_bits(bits: u16) -> Option<u16> {
    let sign = bits & 0x8000;
    let exponent = (bits >> 7) & 0xff;
    let fraction = bits & 0x007f;
    if exponent == 0xff {
        return None;
    }
    if exponent == 0 {
        return Some(sign);
    }

    let unbiased = i32::from(exponent) - 127;
    if unbiased > 15 {
        return None;
    }
    if unbiased >= -14 {
        return Some(sign | (((unbiased + 15) as u16) << 10) | (fraction << 3));
    }

    let significand = 0x0080_u16 | fraction;
    let shift = -(unbiased + 17);
    if shift <= 0 {
        return Some(sign | (significand << (-shift as u32)));
    }
    if shift > 8 {
        return Some(sign);
    }
    let shift = shift as u32;
    let mut rounded = significand >> shift;
    let mask = (1_u16 << shift) - 1;
    let remainder = significand & mask;
    let halfway = 1_u16 << (shift - 1);
    if remainder > halfway || (remainder == halfway && rounded & 1 != 0) {
        rounded += 1;
    }
    Some(sign | rounded)
}

pub fn stream_bf16_payload_to_f16<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
    element_count: u64,
    tensor_name: &str,
) -> Result<Fp16PayloadConversionReport, Fp16CheckpointError> {
    let mut remaining = element_count;
    let mut converted = 0_u64;
    let mut max_buffer_bytes = 0_usize;
    let mut buffer = [0_u8; COPY_BUFFER_BYTES];

    while remaining != 0 {
        let chunk_elements = remaining.min((COPY_BUFFER_BYTES / 2) as u64);
        let chunk_bytes = usize::try_from(chunk_elements * 2)
            .expect("a converter chunk is bounded by COPY_BUFFER_BYTES");
        max_buffer_bytes = max_buffer_bytes.max(chunk_bytes);
        reader
            .read_exact(&mut buffer[..chunk_bytes])
            .map_err(|error| {
                Fp16CheckpointError::tensor_io(
                    error,
                    tensor_name,
                    "failed to read BF16 tensor payload",
                )
            })?;

        for (chunk_index, bytes) in buffer[..chunk_bytes].chunks_exact_mut(2).enumerate() {
            let source = u16::from_le_bytes([bytes[0], bytes[1]]);
            let Some(output) = finite_bf16_to_f16_bits(source) else {
                let exponent = (source >> 7) & 0xff;
                let code = if exponent == 0xff {
                    Fp16CheckpointErrorCode::NonFiniteSource
                } else {
                    Fp16CheckpointErrorCode::OutOfRangeSource
                };
                return Err(Fp16CheckpointError::tensor(
                    code,
                    tensor_name,
                    Some(converted + chunk_index as u64),
                    if exponent == 0xff {
                        "BF16 source value is NaN or infinity"
                    } else {
                        "finite BF16 source value is outside finite F16 range"
                    },
                ));
            };
            bytes.copy_from_slice(&output.to_le_bytes());
        }

        writer.write_all(&buffer[..chunk_bytes]).map_err(|error| {
            Fp16CheckpointError::tensor_io(
                error,
                tensor_name,
                "failed to write converted F16 tensor payload",
            )
        })?;
        converted += chunk_elements;
        remaining -= chunk_elements;
    }

    Ok(Fp16PayloadConversionReport {
        element_count,
        max_buffer_bytes,
    })
}

pub fn convert_bf16_checkpoint_to_f16_stream<R: Read + Seek, W: Write>(
    reader: &mut R,
    source_len: u64,
    writer: &mut W,
) -> Result<Fp16CheckpointConversionReport, Fp16CheckpointError> {
    let source_blake3 = hash_exact_reader(reader, source_len)?;
    reader
        .seek(SeekFrom::Start(0))
        .map_err(|error| Fp16CheckpointError::io(error, "failed to rewind source checkpoint"))?;

    let mut authenticated_reader = HashingReader::new(reader);
    let catalog = SafetensorsCatalog::read_from(&mut authenticated_reader, source_len)?;
    let mut tensors = catalog.tensors().iter().collect::<Vec<_>>();
    tensors.sort_unstable_by_key(|tensor| tensor.data_offsets);
    let mut element_count = 0_u64;
    for tensor in &tensors {
        if tensor.dtype != TensorDtype::BFloat16 {
            return Err(Fp16CheckpointError::tensor(
                Fp16CheckpointErrorCode::UnsupportedSourceDtype,
                &tensor.name,
                None,
                format!(
                    "source tensor dtype is {}, expected BF16",
                    tensor.dtype.safetensors_name()
                ),
            ));
        }
        element_count = element_count
            .checked_add(tensor.byte_len() / 2)
            .ok_or_else(|| {
                Fp16CheckpointError::new(
                    Fp16CheckpointErrorCode::InvalidSource,
                    "checkpoint element count overflowed",
                )
            })?;
    }

    let output_header = output_header(&catalog, &source_blake3)?;
    let mut authenticated_writer = HashingWriter::new(writer);
    authenticated_writer
        .write_all(&(output_header.len() as u64).to_le_bytes())
        .map_err(|error| Fp16CheckpointError::io(error, "failed to write output header length"))?;
    authenticated_writer
        .write_all(&output_header)
        .map_err(|error| Fp16CheckpointError::io(error, "failed to write output header"))?;

    let mut max_payload_buffer_bytes = 0_usize;
    for tensor in tensors {
        let report = stream_bf16_payload_to_f16(
            &mut authenticated_reader,
            &mut authenticated_writer,
            tensor.byte_len() / 2,
            &tensor.name,
        )?;
        max_payload_buffer_bytes = max_payload_buffer_bytes.max(report.max_buffer_bytes);
    }
    authenticated_writer
        .flush()
        .map_err(|error| Fp16CheckpointError::io(error, "failed to flush output checkpoint"))?;

    let (source_hash, source_bytes_read) = authenticated_reader.finish();
    if source_bytes_read != source_len || source_hash != source_blake3 {
        return Err(Fp16CheckpointError::new(
            Fp16CheckpointErrorCode::SourceChanged,
            "source checkpoint changed between identity and conversion passes",
        ));
    }
    let (output_blake3, output_bytes) = authenticated_writer.finish();
    Ok(Fp16CheckpointConversionReport {
        conversion: FP16_CHECKPOINT_CONVERSION_ID,
        tensor_count: catalog.tensors().len() as u64,
        element_count,
        source_bytes: source_len,
        output_bytes,
        source_blake3,
        output_blake3,
        max_payload_buffer_bytes,
    })
}

pub(crate) fn convert_checkpoint_path<F>(
    source: &Path,
    output: &Path,
    convert: F,
) -> Result<Fp16CheckpointConversionReport, Fp16CheckpointError>
where
    F: FnOnce(
        &mut File,
        u64,
        &mut File,
    ) -> Result<Fp16CheckpointConversionReport, Fp16CheckpointError>,
{
    require_distinct_new_output(source, output)?;
    let source_len = fs::metadata(source)
        .map_err(|error| {
            Fp16CheckpointError::io(
                error,
                format!("cannot inspect source checkpoint {}", source.display()),
            )
        })?
        .len();
    let mut source_file = File::open(source).map_err(|error| {
        Fp16CheckpointError::io(
            error,
            format!("cannot open source checkpoint {}", source.display()),
        )
    })?;
    let output_parent = output.parent().unwrap_or_else(|| Path::new("."));
    let mut staged = NamedTempFile::new_in(output_parent).map_err(|error| {
        Fp16CheckpointError::io(
            error,
            format!(
                "cannot create staged checkpoint in {}",
                output_parent.display()
            ),
        )
    })?;
    let report = convert(&mut source_file, source_len, staged.as_file_mut())?;
    staged.as_file_mut().sync_all().map_err(|error| {
        Fp16CheckpointError::io(error, "failed to synchronize staged FP16 checkpoint")
    })?;
    persist_without_overwrite(staged, output)?;
    Ok(report)
}

fn output_header(
    catalog: &SafetensorsCatalog,
    source_blake3: &str,
) -> Result<Vec<u8>, Fp16CheckpointError> {
    let mut metadata = catalog.metadata().clone();
    metadata.insert(
        "pvlc.conversion".to_owned(),
        FP16_CHECKPOINT_CONVERSION_ID.to_owned(),
    );
    metadata.insert("pvlc.source_blake3".to_owned(), source_blake3.to_owned());

    let mut root = Map::new();
    root.insert(
        "__metadata__".to_owned(),
        serde_json::to_value(metadata).map_err(json_error)?,
    );
    for tensor in catalog.tensors() {
        let mut descriptor = Map::new();
        descriptor.insert("dtype".to_owned(), Value::String("F16".to_owned()));
        descriptor.insert(
            "shape".to_owned(),
            serde_json::to_value(&tensor.shape).map_err(json_error)?,
        );
        descriptor.insert(
            "data_offsets".to_owned(),
            serde_json::to_value(tensor.data_offsets).map_err(json_error)?,
        );
        root.insert(tensor.name.clone(), Value::Object(descriptor));
    }
    let mut header = serde_json::to_vec(&Value::Object(root)).map_err(json_error)?;
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    Ok(header)
}

fn json_error(error: serde_json::Error) -> Fp16CheckpointError {
    Fp16CheckpointError::with_source(
        Fp16CheckpointErrorCode::InvalidSource,
        "failed to encode deterministic F16 checkpoint header",
        error,
    )
}

fn hash_exact_reader<R: Read + Seek>(
    reader: &mut R,
    byte_len: u64,
) -> Result<String, Fp16CheckpointError> {
    reader
        .seek(SeekFrom::Start(0))
        .map_err(|error| Fp16CheckpointError::io(error, "failed to seek source checkpoint"))?;
    let mut hasher = blake3::Hasher::new();
    let mut remaining = byte_len;
    let mut buffer = [0_u8; COPY_BUFFER_BYTES];
    while remaining != 0 {
        let requested = usize::try_from(remaining.min(COPY_BUFFER_BYTES as u64))
            .expect("bounded hash chunk fits usize");
        let read = reader
            .read(&mut buffer[..requested])
            .map_err(|error| Fp16CheckpointError::io(error, "failed to hash source checkpoint"))?;
        if read == 0 {
            return Err(Fp16CheckpointError::io(
                io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "source checkpoint is truncated",
                ),
                "failed to hash source checkpoint",
            ));
        }
        hasher.update(&buffer[..read]);
        remaining -= read as u64;
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn require_distinct_new_output(source: &Path, output: &Path) -> Result<(), Fp16CheckpointError> {
    if paths_alias(source, output)? {
        return Err(Fp16CheckpointError::new(
            Fp16CheckpointErrorCode::SourceDestinationAlias,
            "source and output checkpoint paths alias the same file",
        ));
    }
    if output.exists() {
        return Err(Fp16CheckpointError::new(
            Fp16CheckpointErrorCode::OutputAlreadyExists,
            format!("output checkpoint {} already exists", output.display()),
        ));
    }
    Ok(())
}

fn paths_alias(source: &Path, output: &Path) -> Result<bool, Fp16CheckpointError> {
    let source = source.canonicalize().map_err(|error| {
        Fp16CheckpointError::io(
            error,
            format!("cannot resolve source checkpoint {}", source.display()),
        )
    })?;
    if output.exists() {
        let output = output.canonicalize().map_err(|error| {
            Fp16CheckpointError::io(
                error,
                format!("cannot resolve output checkpoint {}", output.display()),
            )
        })?;
        return Ok(source == output);
    }
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let parent = parent.canonicalize().map_err(|error| {
        Fp16CheckpointError::io(
            error,
            format!("cannot resolve output directory {}", parent.display()),
        )
    })?;
    let file_name = output.file_name().ok_or_else(|| {
        Fp16CheckpointError::new(
            Fp16CheckpointErrorCode::InvalidSource,
            "output checkpoint path has no file name",
        )
    })?;
    Ok(source == parent.join(file_name))
}

fn persist_without_overwrite(
    staged: NamedTempFile,
    output: &Path,
) -> Result<(), Fp16CheckpointError> {
    staged
        .persist_noclobber(output)
        .map(|_| ())
        .map_err(|error| {
            if error.error.kind() == io::ErrorKind::AlreadyExists {
                Fp16CheckpointError::new(
                    Fp16CheckpointErrorCode::OutputAlreadyExists,
                    format!("output checkpoint {} already exists", output.display()),
                )
            } else {
                Fp16CheckpointError::io(
                    error.error,
                    format!("failed to persist FP16 checkpoint {}", output.display()),
                )
            }
        })
}

struct HashingReader<'a, R> {
    inner: &'a mut R,
    hasher: blake3::Hasher,
    bytes: u64,
}

impl<'a, R> HashingReader<'a, R> {
    fn new(inner: &'a mut R) -> Self {
        Self {
            inner,
            hasher: blake3::Hasher::new(),
            bytes: 0,
        }
    }

    fn finish(self) -> (String, u64) {
        (self.hasher.finalize().to_hex().to_string(), self.bytes)
    }
}

impl<R: Read> Read for HashingReader<'_, R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let read = self.inner.read(buffer)?;
        self.hasher.update(&buffer[..read]);
        self.bytes += read as u64;
        Ok(read)
    }
}

impl<R: Seek> Seek for HashingReader<'_, R> {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.inner.seek(position)
    }
}

struct HashingWriter<'a, W> {
    inner: &'a mut W,
    hasher: blake3::Hasher,
    bytes: u64,
}

impl<'a, W> HashingWriter<'a, W> {
    fn new(inner: &'a mut W) -> Self {
        Self {
            inner,
            hasher: blake3::Hasher::new(),
            bytes: 0,
        }
    }

    fn finish(self) -> (String, u64) {
        (self.hasher.finalize().to_hex().to_string(), self.bytes)
    }
}

impl<W: Write> Write for HashingWriter<'_, W> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let written = self.inner.write(buffer)?;
        self.hasher.update(&buffer[..written]);
        self.bytes += written as u64;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}
