//! Bounded, header-only safetensors importer.
//!
//! Opening a catalog reads only the eight-byte prefix and the bounded JSON
//! header. Tensor payloads are copied later through a fixed-size buffer, with
//! writer backpressure observed before the next input chunk is read.

mod fp16;

pub use fp16::{
    FP16_CHECKPOINT_CONVERSION_ID, Fp16CheckpointConversionReport, Fp16CheckpointError,
    Fp16CheckpointErrorCode, Fp16PayloadConversionReport, convert_bf16_checkpoint_to_f16_stream,
    finite_bf16_to_f16_bits, stream_bf16_payload_to_f16,
};

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use pvlc_model_schema::{ObservedTensor, PaddleOcrVl16Schema, SchemaError, TensorDtype};
use serde::de::{self, Deserialize, Deserializer, IgnoredAny, MapAccess, Visitor};
use serde_json::Value;

pub const MAX_HEADER_BYTES: u64 = 100_000_000;
pub const COPY_BUFFER_BYTES: usize = 64 * 1_024;

const DUPLICATE_SENTINEL: &str = "PVLC_DUPLICATE_HEADER_KEY";
const INVALID_METADATA_SENTINEL: &str = "PVLC_INVALID_METADATA";
const INVALID_TENSOR_SENTINEL: &str = "PVLC_INVALID_TENSOR_METADATA";

pub fn convert_bf16_checkpoint_to_f16(
    source: impl AsRef<Path>,
    output: impl AsRef<Path>,
) -> Result<Fp16CheckpointConversionReport, Fp16CheckpointError> {
    fp16::convert_checkpoint_path(
        source.as_ref(),
        output.as_ref(),
        |reader, source_len, writer| {
            convert_bf16_checkpoint_to_f16_stream(reader, source_len, writer)
        },
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImportErrorCode {
    Io,
    NotRegularFile,
    HeaderPrefixTruncated,
    HeaderTooLarge,
    HeaderLengthNotAligned,
    HeaderTruncated,
    InvalidHeaderUtf8,
    InvalidHeaderJson,
    DuplicateHeaderKey,
    InvalidMetadata,
    InvalidTensorMetadata,
    UnsupportedDtype,
    UnsupportedTensorConversion,
    TensorMaterializationTooLarge,
    ShapeElementCountOverflow,
    InvalidDataOffsets,
    TensorByteLengthMismatch,
    OverlappingData,
    NonContiguousData,
    DataLengthMismatch,
    TensorNotFound,
    SourceUnavailable,
}

#[derive(Debug)]
pub struct ImportError {
    code: ImportErrorCode,
    tensor_name: Option<String>,
    message: String,
    source: Option<io::Error>,
}

impl ImportError {
    #[must_use]
    pub const fn code(&self) -> ImportErrorCode {
        self.code
    }

    #[must_use]
    pub fn tensor_name(&self) -> Option<&str> {
        self.tensor_name.as_deref()
    }

    fn new(code: ImportErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            tensor_name: None,
            message: message.into(),
            source: None,
        }
    }

    fn tensor(
        code: ImportErrorCode,
        tensor_name: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code,
            tensor_name: Some(tensor_name.into()),
            message: message.into(),
            source: None,
        }
    }

    fn io(error: io::Error, context: impl Into<String>) -> Self {
        Self {
            code: ImportErrorCode::Io,
            tensor_name: None,
            message: context.into(),
            source: Some(error),
        }
    }

    fn tensor_io(error: io::Error, tensor_name: &str, context: impl Into<String>) -> Self {
        Self {
            code: ImportErrorCode::Io,
            tensor_name: Some(tensor_name.to_owned()),
            message: context.into(),
            source: Some(error),
        }
    }
}

impl fmt::Display for ImportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "safetensors import error {:?}: {}",
            self.code, self.message
        )?;
        if let Some(name) = &self.tensor_name {
            write!(formatter, " (tensor {name:?})")?;
        }
        Ok(())
    }
}

impl Error for ImportError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source
            .as_ref()
            .map(|error| error as &(dyn Error + 'static))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorHeader {
    pub name: String,
    pub dtype: TensorDtype,
    pub shape: Vec<u64>,
    /// Byte offsets relative to the beginning of the safetensors data section.
    pub data_offsets: [u64; 2],
}

impl TensorHeader {
    #[must_use]
    pub const fn byte_len(&self) -> u64 {
        self.data_offsets[1] - self.data_offsets[0]
    }
}

#[derive(Clone, Debug)]
pub struct SafetensorsCatalog {
    file_len: u64,
    header_len: u64,
    data_start: u64,
    tensors: Vec<TensorHeader>,
    metadata: BTreeMap<String, String>,
    source_path: Option<PathBuf>,
}

impl SafetensorsCatalog {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ImportError> {
        let path = path.as_ref();
        let (mut file, file_len) = open_regular_file(path)?;
        let mut catalog = Self::read_from(&mut file, file_len)?;
        catalog.source_path = Some(path.to_path_buf());
        Ok(catalog)
    }

    /// Parses only the prefix and header from a reader positioned at byte zero.
    /// `file_len` comes from trusted file metadata and is used to validate every
    /// declared data offset without touching the payload.
    pub fn read_from<R: Read + Seek>(reader: &mut R, file_len: u64) -> Result<Self, ImportError> {
        if file_len < 8 {
            return Err(ImportError::new(
                ImportErrorCode::HeaderPrefixTruncated,
                "file is shorter than the eight-byte header prefix",
            ));
        }

        let mut prefix = [0_u8; 8];
        read_exact_header(reader, &mut prefix, ImportErrorCode::HeaderPrefixTruncated)?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len > MAX_HEADER_BYTES {
            return Err(ImportError::new(
                ImportErrorCode::HeaderTooLarge,
                format!("declared header is {header_len} bytes"),
            ));
        }
        if !header_len.is_multiple_of(8) {
            return Err(ImportError::new(
                ImportErrorCode::HeaderLengthNotAligned,
                format!("declared header length {header_len} is not eight-byte aligned"),
            ));
        }
        let data_start = 8_u64.checked_add(header_len).ok_or_else(|| {
            ImportError::new(ImportErrorCode::HeaderTooLarge, "header offset overflow")
        })?;
        if data_start > file_len {
            return Err(ImportError::new(
                ImportErrorCode::HeaderTruncated,
                "declared header extends beyond the file",
            ));
        }

        let header_size = usize::try_from(header_len).map_err(|_| {
            ImportError::new(
                ImportErrorCode::HeaderTooLarge,
                "header cannot fit in address space",
            )
        })?;
        let mut header_bytes = vec![0_u8; header_size];
        read_exact_header(reader, &mut header_bytes, ImportErrorCode::HeaderTruncated)?;
        let header_text = std::str::from_utf8(&header_bytes).map_err(|error| {
            ImportError::new(
                ImportErrorCode::InvalidHeaderUtf8,
                format!("header is not UTF-8: {error}"),
            )
        })?;
        let raw: RawHeader = serde_json::from_str(header_text).map_err(classify_json_error)?;

        let body_len = file_len - data_start;
        let mut tensors = Vec::with_capacity(raw.tensors.len());
        for (name, raw_tensor) in raw.tensors {
            tensors.push(validate_tensor(name, raw_tensor)?);
        }
        validate_data_layout(&tensors, body_len)?;
        tensors.sort_unstable_by(|left, right| left.name.cmp(&right.name));

        Ok(Self {
            file_len,
            header_len,
            data_start,
            tensors,
            metadata: raw.metadata,
            source_path: None,
        })
    }

    #[must_use]
    pub const fn file_len(&self) -> u64 {
        self.file_len
    }

    #[must_use]
    pub const fn header_len(&self) -> u64 {
        self.header_len
    }

    #[must_use]
    pub fn tensors(&self) -> &[TensorHeader] {
        &self.tensors
    }

    #[must_use]
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&TensorHeader> {
        self.tensors
            .binary_search_by(|tensor| tensor.name.as_str().cmp(name))
            .ok()
            .map(|index| &self.tensors[index])
    }

    pub fn validate_paddleocr_vl_16(&self) -> Result<(), SchemaError> {
        let observed: Vec<_> = self
            .tensors
            .iter()
            .map(|tensor| ObservedTensor::new(&tensor.name, tensor.dtype, tensor.shape.clone()))
            .collect();
        PaddleOcrVl16Schema::validate(&observed)
    }

    pub fn copy_tensor_to<W: Write>(&self, name: &str, writer: &mut W) -> Result<u64, ImportError> {
        let path = self.source_path.as_ref().ok_or_else(|| {
            ImportError::new(
                ImportErrorCode::SourceUnavailable,
                "catalog was created from a borrowed reader",
            )
        })?;
        let (mut file, file_len) = open_regular_file(path)?;
        if file_len != self.file_len {
            return Err(ImportError::new(
                ImportErrorCode::DataLengthMismatch,
                "source file length changed after catalog import",
            ));
        }
        self.copy_tensor_from(name, &mut file, writer)
    }

    /// Materializes a BF16, F16, or F32 tensor as native `f32` values while
    /// preserving the exact stored bit pattern of every source element.
    pub fn load_tensor_f32(&self, name: &str) -> Result<Vec<f32>, ImportError> {
        let tensor = self.tensor(name).ok_or_else(|| {
            ImportError::tensor(
                ImportErrorCode::TensorNotFound,
                name,
                "tensor is absent from the catalog",
            )
        })?;
        let element_bytes = match tensor.dtype {
            TensorDtype::BFloat16 | TensorDtype::Float16 => 2,
            TensorDtype::Float32 => 4,
            _ => {
                return Err(ImportError::tensor(
                    ImportErrorCode::UnsupportedTensorConversion,
                    name,
                    format!(
                        "cannot materialize {} as f32 values",
                        tensor.dtype.safetensors_name()
                    ),
                ));
            }
        };
        let byte_len = usize::try_from(tensor.byte_len()).map_err(|_| {
            ImportError::tensor(
                ImportErrorCode::TensorMaterializationTooLarge,
                name,
                "tensor payload cannot fit in address space",
            )
        })?;
        let element_count = byte_len / element_bytes;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(byte_len).map_err(|_| {
            ImportError::tensor(
                ImportErrorCode::TensorMaterializationTooLarge,
                name,
                "tensor payload allocation failed",
            )
        })?;
        self.copy_tensor_to(name, &mut bytes)?;

        let mut values = Vec::new();
        values.try_reserve_exact(element_count).map_err(|_| {
            ImportError::tensor(
                ImportErrorCode::TensorMaterializationTooLarge,
                name,
                "converted tensor allocation failed",
            )
        })?;
        match tensor.dtype {
            TensorDtype::BFloat16 => values.extend(bytes.chunks_exact(2).map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                f32::from_bits(u32::from(bits) << 16)
            })),
            TensorDtype::Float16 => values.extend(bytes.chunks_exact(2).map(|chunk| {
                f32::from_bits(f16_to_f32_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
            })),
            TensorDtype::Float32 => values.extend(bytes.chunks_exact(4).map(|chunk| {
                f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            })),
            _ => unreachable!("conversion dtype was validated above"),
        }
        Ok(values)
    }

    pub fn copy_tensor_from<R: Read + Seek, W: Write>(
        &self,
        name: &str,
        reader: &mut R,
        writer: &mut W,
    ) -> Result<u64, ImportError> {
        let tensor = self.tensor(name).ok_or_else(|| {
            ImportError::tensor(
                ImportErrorCode::TensorNotFound,
                name,
                "tensor is absent from the catalog",
            )
        })?;
        let absolute_start = self
            .data_start
            .checked_add(tensor.data_offsets[0])
            .ok_or_else(|| {
                ImportError::tensor(
                    ImportErrorCode::InvalidDataOffsets,
                    name,
                    "absolute tensor offset overflow",
                )
            })?;
        reader
            .seek(SeekFrom::Start(absolute_start))
            .map_err(|error| {
                ImportError::tensor_io(error, name, "failed to seek to tensor payload")
            })?;

        let mut remaining = tensor.byte_len();
        let mut copied = 0_u64;
        let mut buffer = [0_u8; COPY_BUFFER_BYTES];
        while remaining != 0 {
            let chunk_len = usize::try_from(remaining.min(COPY_BUFFER_BYTES as u64))
                .expect("bounded chunk always fits usize");
            reader
                .read_exact(&mut buffer[..chunk_len])
                .map_err(|error| {
                    ImportError::tensor_io(error, name, "failed to read tensor payload")
                })?;
            // Complete this write before reading the next input chunk. Besides
            // bounding memory, this propagates downstream backpressure.
            writer.write_all(&buffer[..chunk_len]).map_err(|error| {
                ImportError::tensor_io(error, name, "failed to write tensor payload")
            })?;
            let chunk_len = chunk_len as u64;
            remaining -= chunk_len;
            copied += chunk_len;
        }
        Ok(copied)
    }
}

fn f16_to_f32_bits(bits: u16) -> u32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mut fraction = bits & 0x03ff;
    match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let mut unbiased = -14_i32;
            while fraction & 0x0400 == 0 {
                fraction <<= 1;
                unbiased -= 1;
            }
            fraction &= 0x03ff;
            sign | (((unbiased + 127) as u32) << 23) | (u32::from(fraction) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (u32::from(fraction) << 13),
        _ => sign | (u32::from(exponent - 15 + 127) << 23) | (u32::from(fraction) << 13),
    }
}

fn read_exact_header<R: Read>(
    reader: &mut R,
    buffer: &mut [u8],
    truncated_code: ImportErrorCode,
) -> Result<(), ImportError> {
    reader.read_exact(buffer).map_err(|error| {
        if error.kind() == io::ErrorKind::UnexpectedEof {
            ImportError::new(truncated_code, "safetensors header is truncated")
        } else {
            ImportError::io(error, "failed to read safetensors header")
        }
    })
}

fn open_regular_file(path: &Path) -> Result<(File, u64), ImportError> {
    let before = fs::symlink_metadata(path)
        .map_err(|error| ImportError::io(error, format!("cannot inspect {}", path.display())))?;
    if !before.file_type().is_file() {
        return Err(ImportError::new(
            ImportErrorCode::NotRegularFile,
            format!("{} is not a regular file", path.display()),
        ));
    }
    let file = File::open(path)
        .map_err(|error| ImportError::io(error, format!("cannot open {}", path.display())))?;
    let after = file
        .metadata()
        .map_err(|error| ImportError::io(error, format!("cannot stat {}", path.display())))?;
    if !after.file_type().is_file() || !same_file(&before, &after) {
        return Err(ImportError::new(
            ImportErrorCode::NotRegularFile,
            format!("{} changed while it was being opened", path.display()),
        ));
    }
    Ok((file, after.len()))
}

#[cfg(unix)]
fn same_file(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    left.dev() == right.dev() && left.ino() == right.ino()
}

#[cfg(not(unix))]
fn same_file(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    left.len() == right.len() && left.modified().ok() == right.modified().ok()
}

fn classify_json_error(error: serde_json::Error) -> ImportError {
    let message = error.to_string();
    let code = if message.contains(DUPLICATE_SENTINEL) {
        ImportErrorCode::DuplicateHeaderKey
    } else if message.contains(INVALID_METADATA_SENTINEL) {
        ImportErrorCode::InvalidMetadata
    } else if message.contains(INVALID_TENSOR_SENTINEL) {
        ImportErrorCode::InvalidTensorMetadata
    } else {
        ImportErrorCode::InvalidHeaderJson
    };
    ImportError::new(code, message)
}

fn validate_tensor(name: String, raw: RawTensor) -> Result<TensorHeader, ImportError> {
    if name.is_empty() {
        return Err(ImportError::tensor(
            ImportErrorCode::InvalidTensorMetadata,
            name,
            "tensor name is empty",
        ));
    }
    let dtype = parse_dtype(&raw.dtype).ok_or_else(|| {
        ImportError::tensor(
            ImportErrorCode::UnsupportedDtype,
            &name,
            format!("unsupported dtype {:?}", raw.dtype),
        )
    })?;
    let element_count = raw
        .shape
        .iter()
        .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension));
    let element_count = element_count.ok_or_else(|| {
        ImportError::tensor(
            ImportErrorCode::ShapeElementCountOverflow,
            &name,
            "shape element count overflows u64",
        )
    })?;
    let expected_bytes = element_count
        .checked_mul(dtype.byte_width())
        .ok_or_else(|| {
            ImportError::tensor(
                ImportErrorCode::ShapeElementCountOverflow,
                &name,
                "tensor byte count overflows u64",
            )
        })?;
    let [start, end] = raw.data_offsets;
    let actual_bytes = end.checked_sub(start).ok_or_else(|| {
        ImportError::tensor(
            ImportErrorCode::InvalidDataOffsets,
            &name,
            "tensor data offsets are reversed",
        )
    })?;
    if actual_bytes != expected_bytes {
        return Err(ImportError::tensor(
            ImportErrorCode::TensorByteLengthMismatch,
            &name,
            format!("shape requires {expected_bytes} bytes but offsets span {actual_bytes}"),
        ));
    }
    Ok(TensorHeader {
        name,
        dtype,
        shape: raw.shape,
        data_offsets: raw.data_offsets,
    })
}

fn validate_data_layout(tensors: &[TensorHeader], body_len: u64) -> Result<(), ImportError> {
    let mut by_offset: Vec<_> = tensors.iter().collect();
    by_offset.sort_unstable_by_key(|tensor| {
        (
            tensor.data_offsets[0],
            tensor.data_offsets[1],
            tensor.name.as_str(),
        )
    });
    let mut cursor = 0_u64;
    for tensor in by_offset {
        let start = tensor.data_offsets[0];
        if start < cursor {
            return Err(ImportError::tensor(
                ImportErrorCode::OverlappingData,
                &tensor.name,
                "tensor payload overlaps a previous tensor",
            ));
        }
        if start > cursor {
            return Err(ImportError::tensor(
                ImportErrorCode::NonContiguousData,
                &tensor.name,
                "tensor payload leaves an unclaimed gap",
            ));
        }
        cursor = tensor.data_offsets[1];
    }
    if cursor != body_len {
        return Err(ImportError::new(
            ImportErrorCode::DataLengthMismatch,
            format!("tensor data covers {cursor} bytes but file body is {body_len} bytes"),
        ));
    }
    Ok(())
}

fn parse_dtype(value: &str) -> Option<TensorDtype> {
    Some(match value {
        "BOOL" => TensorDtype::Bool,
        "U8" => TensorDtype::Uint8,
        "I8" => TensorDtype::Int8,
        "U16" => TensorDtype::Uint16,
        "I16" => TensorDtype::Int16,
        "U32" => TensorDtype::Uint32,
        "I32" => TensorDtype::Int32,
        "U64" => TensorDtype::Uint64,
        "I64" => TensorDtype::Int64,
        "F8_E4M3" => TensorDtype::Float8E4M3,
        "F8_E5M2" => TensorDtype::Float8E5M2,
        "BF16" => TensorDtype::BFloat16,
        "F16" => TensorDtype::Float16,
        "F32" => TensorDtype::Float32,
        "F64" => TensorDtype::Float64,
        _ => return None,
    })
}

#[derive(Debug)]
struct RawHeader {
    metadata: BTreeMap<String, String>,
    tensors: Vec<(String, RawTensor)>,
}

impl<'de> Deserialize<'de> for RawHeader {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_map(RawHeaderVisitor)
    }
}

struct RawHeaderVisitor;

impl<'de> Visitor<'de> for RawHeaderVisitor {
    type Value = RawHeader;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a safetensors header object")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut seen = BTreeSet::new();
        let mut metadata = None;
        let mut tensors = Vec::new();
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(de::Error::custom(format!("{DUPLICATE_SENTINEL}: {key}")));
            }
            if key == "__metadata__" {
                metadata = Some(map.next_value::<RawMetadata>()?.0);
            } else {
                tensors.push((key, map.next_value::<RawTensor>()?));
            }
        }
        Ok(RawHeader {
            metadata: metadata.unwrap_or_default(),
            tensors,
        })
    }
}

struct RawMetadata(BTreeMap<String, String>);

impl<'de> Deserialize<'de> for RawMetadata {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_map(RawMetadataVisitor)
    }
}

struct RawMetadataVisitor;

impl<'de> Visitor<'de> for RawMetadataVisitor {
    type Value = RawMetadata;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a string-to-string metadata object")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut values = BTreeMap::new();
        while let Some(key) = map.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(de::Error::custom(format!(
                    "{DUPLICATE_SENTINEL}: metadata.{key}"
                )));
            }
            let value = map.next_value::<Value>()?;
            let Value::String(value) = value else {
                return Err(de::Error::custom(format!(
                    "{INVALID_METADATA_SENTINEL}: metadata.{key} is not a string"
                )));
            };
            values.insert(key, value);
        }
        Ok(RawMetadata(values))
    }
}

#[derive(Debug)]
struct RawTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

impl<'de> Deserialize<'de> for RawTensor {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_map(RawTensorVisitor)
    }
}

struct RawTensorVisitor;

impl<'de> Visitor<'de> for RawTensorVisitor {
    type Value = RawTensor;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a tensor metadata object")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut seen = BTreeSet::new();
        let mut dtype = None;
        let mut shape = None;
        let mut data_offsets = None;
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(de::Error::custom(format!(
                    "{DUPLICATE_SENTINEL}: tensor field {key}"
                )));
            }
            let value = match key.as_str() {
                "dtype" | "shape" | "data_offsets" => map.next_value::<Value>()?,
                _ => {
                    map.next_value::<IgnoredAny>()?;
                    return Err(de::Error::custom(format!(
                        "{INVALID_TENSOR_SENTINEL}: unknown field {key}"
                    )));
                }
            };
            match key.as_str() {
                "dtype" => dtype = Some(value),
                "shape" => shape = Some(value),
                "data_offsets" => data_offsets = Some(value),
                _ => unreachable!("handled above"),
            }
        }

        let dtype = value_string::<A::Error>(dtype, "dtype")?;
        let shape = value_u64_array::<A::Error>(shape, "shape")?;
        let offsets = value_u64_array::<A::Error>(data_offsets, "data_offsets")?;
        let data_offsets: [u64; 2] = offsets.try_into().map_err(|_| {
            de::Error::custom(format!(
                "{INVALID_TENSOR_SENTINEL}: data_offsets must contain two integers"
            ))
        })?;
        Ok(RawTensor {
            dtype,
            shape,
            data_offsets,
        })
    }
}

fn value_string<E: de::Error>(value: Option<Value>, field: &str) -> Result<String, E> {
    match value {
        Some(Value::String(value)) => Ok(value),
        Some(_) => Err(E::custom(format!(
            "{INVALID_TENSOR_SENTINEL}: {field} has the wrong type"
        ))),
        None => Err(E::custom(format!(
            "{INVALID_TENSOR_SENTINEL}: missing {field}"
        ))),
    }
}

fn value_u64_array<E: de::Error>(value: Option<Value>, field: &str) -> Result<Vec<u64>, E> {
    let Some(Value::Array(values)) = value else {
        return Err(E::custom(format!(
            "{INVALID_TENSOR_SENTINEL}: {field} must be an array"
        )));
    };
    values
        .into_iter()
        .map(|value| {
            value.as_u64().ok_or_else(|| {
                E::custom(format!(
                    "{INVALID_TENSOR_SENTINEL}: {field} must contain unsigned integers"
                ))
            })
        })
        .collect()
}
