//! Offline compiler source-integrity gate and deterministic tiny-pack path.

mod vision_stack;

pub use vision_stack::*;

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

use pvlc_ir::SemanticGraph;
use pvlc_model_schema::{COMPILER_MODEL_ABI, MODEL_ID, MODEL_REVISION, PaddleOcrVl16Schema};
use pvlc_pack::{PackBuilder, PackManifest, PackSection, PrecisionProfile, SectionKind};
use serde::{Deserialize, Serialize};

const MODEL_LOCK_FORMAT_VERSION: u32 = 1;
const MAX_MODEL_LOCK_BYTES: u64 = 1_048_576;
const HASH_BUFFER_BYTES: usize = 256 * 1_024;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawModelLock {
    format_version: u32,
    model_id: String,
    revision: String,
    compiler_model_abi: u32,
    files: BTreeMap<String, LockedFile>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LockedFile {
    pub blake3: String,
    pub size: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelLock {
    pub format_version: u32,
    pub model_id: String,
    pub revision: String,
    pub compiler_model_abi: u32,
    files: BTreeMap<String, LockedFile>,
    raw_blake3: [u8; 32],
}

impl ModelLock {
    pub fn parse(bytes: &[u8]) -> Result<Self, SourceError> {
        if bytes.len() as u64 > MAX_MODEL_LOCK_BYTES {
            return Err(SourceError::new(
                SourceErrorCode::InvalidLock,
                "model lock exceeds its size bound",
            ));
        }
        let text = std::str::from_utf8(bytes).map_err(|_| {
            SourceError::new(SourceErrorCode::InvalidLock, "model lock is not UTF-8")
        })?;
        let raw: RawModelLock = toml::from_str(text).map_err(|error| {
            SourceError::new(
                SourceErrorCode::InvalidLock,
                format!("invalid model lock TOML: {error}"),
            )
        })?;
        if raw.format_version != MODEL_LOCK_FORMAT_VERSION {
            return Err(SourceError::new(
                SourceErrorCode::UnsupportedFormatVersion,
                format!("unsupported model lock version {}", raw.format_version),
            ));
        }
        if raw.files.is_empty() {
            return Err(SourceError::new(
                SourceErrorCode::EmptyFileSet,
                "model lock contains no files",
            ));
        }
        for (path, file) in &raw.files {
            validate_locked_path(path)?;
            if !is_lower_hex_64(&file.blake3) {
                return Err(SourceError::at_path(
                    SourceErrorCode::InvalidHash,
                    path,
                    "BLAKE3 digest must be exactly 64 lowercase hexadecimal characters",
                ));
            }
        }
        Ok(Self {
            format_version: raw.format_version,
            model_id: raw.model_id,
            revision: raw.revision,
            compiler_model_abi: raw.compiler_model_abi,
            files: raw.files,
            raw_blake3: *blake3::hash(bytes).as_bytes(),
        })
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, SourceError> {
        let path = path.as_ref();
        let metadata = fs::symlink_metadata(path).map_err(|error| {
            SourceError::io(
                error,
                format!("cannot inspect model lock {}", path.display()),
            )
        })?;
        if !metadata.file_type().is_file() {
            return Err(SourceError::new(
                SourceErrorCode::NotRegularFile,
                format!("model lock {} is not a regular file", path.display()),
            ));
        }
        if metadata.len() > MAX_MODEL_LOCK_BYTES {
            return Err(SourceError::new(
                SourceErrorCode::InvalidLock,
                "model lock exceeds its size bound",
            ));
        }
        let bytes = fs::read(path).map_err(|error| {
            SourceError::io(error, format!("cannot read model lock {}", path.display()))
        })?;
        Self::parse(&bytes)
    }

    #[must_use]
    pub fn files(&self) -> &BTreeMap<String, LockedFile> {
        &self.files
    }

    #[must_use]
    pub fn file(&self, path: &str) -> Option<&LockedFile> {
        self.files.get(path)
    }

    #[must_use]
    pub const fn raw_blake3(&self) -> [u8; 32] {
        self.raw_blake3
    }

    #[must_use]
    pub fn raw_blake3_hex(&self) -> String {
        blake3::Hash::from_bytes(self.raw_blake3)
            .to_hex()
            .to_string()
    }

    fn validate_identity(&self) -> Result<(), SourceError> {
        if self.model_id != MODEL_ID {
            return Err(SourceError::new(
                SourceErrorCode::WrongModelId,
                format!("compiler supports {MODEL_ID:?}, not {:?}", self.model_id),
            ));
        }
        if self.revision != MODEL_REVISION {
            return Err(SourceError::new(
                SourceErrorCode::WrongModelRevision,
                format!(
                    "compiler supports revision {MODEL_REVISION}, not {}",
                    self.revision
                ),
            ));
        }
        if self.compiler_model_abi != COMPILER_MODEL_ABI {
            return Err(SourceError::new(
                SourceErrorCode::WrongCompilerModelAbi,
                format!(
                    "compiler supports model ABI {COMPILER_MODEL_ABI}, not {}",
                    self.compiler_model_abi
                ),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedFile {
    pub path: String,
    pub size: u64,
    pub blake3: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedModelSource {
    pub lock_blake3: [u8; 32],
    pub source_fingerprint: [u8; 32],
    pub files: Vec<VerifiedFile>,
}

pub fn verify_model_source(
    model_lock: &ModelLock,
    model_dir: impl AsRef<Path>,
) -> Result<VerifiedModelSource, SourceError> {
    // Identity is checked before even statting a user-supplied directory.
    model_lock.validate_identity()?;
    let model_dir = model_dir.as_ref();
    let root_metadata = fs::symlink_metadata(model_dir).map_err(|error| {
        SourceError::io(
            error,
            format!("cannot inspect model directory {}", model_dir.display()),
        )
    })?;
    if !root_metadata.file_type().is_dir() {
        return Err(SourceError::new(
            SourceErrorCode::NotRegularFile,
            format!("{} is not a real directory", model_dir.display()),
        ));
    }

    let mut verified_files = Vec::with_capacity(model_lock.files.len());
    for (relative_path, locked) in &model_lock.files {
        let full_path = validate_path_components(model_dir, relative_path)?;
        let (mut file, metadata_before) = open_expected_regular(&full_path, relative_path)?;
        if metadata_before.len() != locked.size {
            return Err(SourceError::at_path(
                SourceErrorCode::SizeMismatch,
                relative_path,
                format!(
                    "lock expects {} bytes but file has {}",
                    locked.size,
                    metadata_before.len()
                ),
            ));
        }
        let mut hasher = blake3::Hasher::new();
        let mut total = 0_u64;
        let mut buffer = [0_u8; HASH_BUFFER_BYTES];
        loop {
            let read = file.read(&mut buffer).map_err(|error| {
                SourceError::path_io(error, relative_path, "failed while hashing model file")
            })?;
            if read == 0 {
                break;
            }
            total = total.checked_add(read as u64).ok_or_else(|| {
                SourceError::at_path(
                    SourceErrorCode::SizeMismatch,
                    relative_path,
                    "file size overflow while hashing",
                )
            })?;
            hasher.update(&buffer[..read]);
        }
        let metadata_after = file.metadata().map_err(|error| {
            SourceError::path_io(error, relative_path, "cannot restat hashed model file")
        })?;
        if total != locked.size || !same_snapshot(&metadata_before, &metadata_after) {
            return Err(SourceError::at_path(
                SourceErrorCode::SizeMismatch,
                relative_path,
                "model file changed while it was being hashed",
            ));
        }
        let actual_hash = hasher.finalize();
        if actual_hash.to_hex().as_str() != locked.blake3 {
            return Err(SourceError::at_path(
                SourceErrorCode::HashMismatch,
                relative_path,
                "model file BLAKE3 does not match model.lock",
            ));
        }
        verified_files.push(VerifiedFile {
            path: relative_path.clone(),
            size: total,
            blake3: *actual_hash.as_bytes(),
        });
    }

    let discovered = discover_regular_files(model_dir)?;
    let expected: BTreeSet<_> = model_lock.files.keys().cloned().collect();
    if let Some(unexpected) = discovered.difference(&expected).next() {
        return Err(SourceError::at_path(
            SourceErrorCode::UnexpectedFile,
            unexpected,
            "file is not listed in model.lock",
        ));
    }
    if let Some(missing) = expected.difference(&discovered).next() {
        return Err(SourceError::at_path(
            SourceErrorCode::MissingFile,
            missing,
            "locked file is absent",
        ));
    }

    let source_fingerprint = canonical_source_fingerprint(model_lock)?;
    Ok(VerifiedModelSource {
        lock_blake3: model_lock.raw_blake3,
        source_fingerprint,
        files: verified_files,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompileTinyOptions {
    pub compiler_build: String,
    pub precision_profile: PrecisionProfile,
    pub resolution_buckets: Vec<[u32; 2]>,
    pub context_limit: u32,
}

pub fn compile_tiny_pack(
    model_lock: &ModelLock,
    model_dir: impl AsRef<Path>,
    options: &CompileTinyOptions,
) -> Result<Vec<u8>, SourceError> {
    if !is_lower_hex_64(&options.compiler_build) {
        return Err(SourceError::new(
            SourceErrorCode::InvalidCompilerBuild,
            "compiler build must be exactly 64 lowercase hexadecimal characters",
        ));
    }
    let verified = verify_model_source(model_lock, model_dir)?;
    let manifest = PackManifest {
        model_id: MODEL_ID.to_owned(),
        model_revision: MODEL_REVISION.to_owned(),
        compiler_model_abi: COMPILER_MODEL_ABI,
        compiler_build: options.compiler_build.clone(),
        precision_profile: options.precision_profile,
        resolution_buckets: options.resolution_buckets.clone(),
        context_limit: options.context_limit,
    };
    let mut builder = PackBuilder::new(manifest);
    builder.add_section(PackSection::new(
        "model.schema",
        SectionKind::ModelSchema,
        64,
        PaddleOcrVl16Schema::canonical_catalog_bytes(),
    ))?;
    builder.add_section(PackSection::new(
        "model.semantic_map",
        SectionKind::SemanticMap,
        64,
        PaddleOcrVl16Schema::canonical_semantic_map_bytes(),
    ))?;
    builder.add_section(PackSection::new(
        "ir.semantic",
        SectionKind::SemanticIr,
        64,
        SemanticGraph::paddleocr_vl_16().canonical_bytes()?,
    ))?;
    let mut provenance = Vec::with_capacity(64);
    provenance.extend_from_slice(&verified.lock_blake3);
    provenance.extend_from_slice(&verified.source_fingerprint);
    builder.add_section(PackSection::new(
        "self_test.source_provenance",
        SectionKind::SelfTest,
        32,
        provenance,
    ))?;
    builder.build().map_err(SourceError::from)
}

pub fn compile_tiny_to_path(
    lock_path: impl AsRef<Path>,
    model_dir: impl AsRef<Path>,
    output: impl AsRef<Path>,
    options: &CompileTinyOptions,
) -> Result<(), SourceError> {
    let lock = ModelLock::from_path(lock_path)?;
    let bytes = compile_tiny_pack(&lock, model_dir, options)?;
    write_atomic(output.as_ref(), &bytes)
}

fn write_atomic(output: &Path, bytes: &[u8]) -> Result<(), SourceError> {
    match fs::symlink_metadata(output) {
        Ok(metadata) if !metadata.file_type().is_file() => {
            return Err(SourceError::new(
                SourceErrorCode::InvalidOutput,
                format!("output {} is not a regular file", output.display()),
            ));
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(SourceError::io(
                error,
                format!("cannot inspect output {}", output.display()),
            ));
        }
    }
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let parent_metadata = fs::symlink_metadata(parent).map_err(|error| {
        SourceError::io(
            error,
            format!("cannot inspect output directory {}", parent.display()),
        )
    })?;
    if !parent_metadata.file_type().is_dir() {
        return Err(SourceError::new(
            SourceErrorCode::InvalidOutput,
            format!("output parent {} is not a real directory", parent.display()),
        ));
    }

    let mut temporary = tempfile::NamedTempFile::new_in(parent).map_err(|error| {
        SourceError::io(
            error,
            format!("cannot create staged output in {}", parent.display()),
        )
    })?;
    temporary.write_all(bytes).map_err(|error| {
        SourceError::io(
            error,
            format!("cannot write staged output {}", output.display()),
        )
    })?;
    temporary.as_file_mut().sync_all().map_err(|error| {
        SourceError::io(
            error,
            format!("cannot sync staged output {}", output.display()),
        )
    })?;
    temporary.persist(output).map_err(|error| {
        SourceError::io(
            error.error,
            format!("cannot atomically install output {}", output.display()),
        )
    })?;
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| {
            SourceError::io(
                error,
                format!("cannot sync output directory {}", parent.display()),
            )
        })?;
    Ok(())
}

fn canonical_source_fingerprint(model_lock: &ModelLock) -> Result<[u8; 32], SourceError> {
    #[derive(Serialize)]
    struct CanonicalFile<'a> {
        blake3: &'a str,
        path: &'a str,
        size: u64,
    }
    #[derive(Serialize)]
    struct CanonicalSource<'a> {
        compiler_model_abi: u32,
        files: Vec<CanonicalFile<'a>>,
        model_id: &'a str,
        revision: &'a str,
    }
    let files = model_lock
        .files
        .iter()
        .map(|(path, file)| CanonicalFile {
            blake3: &file.blake3,
            path,
            size: file.size,
        })
        .collect();
    let source = CanonicalSource {
        compiler_model_abi: model_lock.compiler_model_abi,
        files,
        model_id: &model_lock.model_id,
        revision: &model_lock.revision,
    };
    let mut bytes = serde_json::to_vec(&source).map_err(|error| {
        SourceError::new(
            SourceErrorCode::InvalidLock,
            format!("cannot canonicalize model lock: {error}"),
        )
    })?;
    bytes.push(b'\n');
    Ok(*blake3::hash(&bytes).as_bytes())
}

fn validate_locked_path(path: &str) -> Result<(), SourceError> {
    if path.is_empty() || path.contains('\\') || path.contains(':') {
        return Err(SourceError::at_path(
            SourceErrorCode::UnsafePath,
            path,
            "locked path is empty or not portable",
        ));
    }
    let parsed = Path::new(path);
    if parsed.is_absolute()
        || parsed
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(SourceError::at_path(
            SourceErrorCode::UnsafePath,
            path,
            "locked path must contain only normal relative components",
        ));
    }
    Ok(())
}

fn validate_path_components(root: &Path, relative_path: &str) -> Result<PathBuf, SourceError> {
    let mut current = root.to_path_buf();
    let components: Vec<_> = Path::new(relative_path).components().collect();
    for (index, component) in components.iter().enumerate() {
        let Component::Normal(component) = component else {
            unreachable!("lock paths were validated while parsing")
        };
        current.push(component);
        let metadata = match fs::symlink_metadata(&current) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(SourceError::at_path(
                    SourceErrorCode::MissingFile,
                    relative_path,
                    "locked file or parent directory is absent",
                ));
            }
            Err(error) => {
                return Err(SourceError::path_io(
                    error,
                    relative_path,
                    "cannot inspect locked path",
                ));
            }
        };
        let is_last = index + 1 == components.len();
        let expected_type = if is_last {
            metadata.file_type().is_file()
        } else {
            metadata.file_type().is_dir()
        };
        if !expected_type {
            return Err(SourceError::at_path(
                SourceErrorCode::NotRegularFile,
                relative_path,
                "locked path contains a symlink or non-regular component",
            ));
        }
    }
    Ok(current)
}

fn open_expected_regular(
    path: &Path,
    relative_path: &str,
) -> Result<(File, fs::Metadata), SourceError> {
    let before = fs::symlink_metadata(path).map_err(|error| {
        SourceError::path_io(error, relative_path, "cannot inspect locked file")
    })?;
    if !before.file_type().is_file() {
        return Err(SourceError::at_path(
            SourceErrorCode::NotRegularFile,
            relative_path,
            "locked path is not a regular file",
        ));
    }
    let file = File::open(path)
        .map_err(|error| SourceError::path_io(error, relative_path, "cannot open locked file"))?;
    let opened = file.metadata().map_err(|error| {
        SourceError::path_io(error, relative_path, "cannot stat opened locked file")
    })?;
    if !opened.file_type().is_file() || !same_file(&before, &opened) {
        return Err(SourceError::at_path(
            SourceErrorCode::NotRegularFile,
            relative_path,
            "locked file changed while it was being opened",
        ));
    }
    Ok((file, opened))
}

fn discover_regular_files(root: &Path) -> Result<BTreeSet<String>, SourceError> {
    fn visit(
        root: &Path,
        directory: &Path,
        output: &mut BTreeSet<String>,
    ) -> Result<(), SourceError> {
        let mut entries: Vec<_> = fs::read_dir(directory)
            .map_err(|error| {
                SourceError::io(
                    error,
                    format!("cannot enumerate model directory {}", directory.display()),
                )
            })?
            .collect::<Result<_, _>>()
            .map_err(|error| SourceError::io(error, "cannot enumerate model directory entry"))?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            let relative = path.strip_prefix(root).expect("entry is below root");
            let relative = relative
                .components()
                .map(|component| component.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/");
            let file_type = entry.file_type().map_err(|error| {
                SourceError::path_io(error, &relative, "cannot inspect model directory entry")
            })?;
            if file_type.is_dir() {
                visit(root, &path, output)?;
            } else if file_type.is_file() {
                output.insert(relative);
            } else {
                return Err(SourceError::at_path(
                    SourceErrorCode::NotRegularFile,
                    relative,
                    "model directory contains a symlink or special file",
                ));
            }
        }
        Ok(())
    }

    let mut output = BTreeSet::new();
    visit(root, root, &mut output)?;
    Ok(output)
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

#[cfg(unix)]
fn same_snapshot(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    same_file(left, right)
        && left.len() == right.len()
        && left.mtime() == right.mtime()
        && left.mtime_nsec() == right.mtime_nsec()
}

#[cfg(not(unix))]
fn same_snapshot(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    same_file(left, right)
        && left.len() == right.len()
        && left.modified().ok() == right.modified().ok()
}

fn is_lower_hex_64(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceErrorCode {
    InvalidLock,
    UnsupportedFormatVersion,
    InvalidHash,
    UnsafePath,
    EmptyFileSet,
    WrongModelId,
    WrongModelRevision,
    WrongCompilerModelAbi,
    MissingFile,
    UnexpectedFile,
    NotRegularFile,
    SizeMismatch,
    HashMismatch,
    InvalidCompilerBuild,
    InvalidOutput,
    GoldenMismatch,
    Safetensors,
    VisionStackShard,
    Pack,
    Io,
}

#[derive(Debug)]
pub struct SourceError {
    code: SourceErrorCode,
    path: Option<String>,
    message: String,
    source: Option<Box<dyn Error + Send + Sync>>,
}

impl SourceError {
    #[must_use]
    pub const fn code(&self) -> SourceErrorCode {
        self.code
    }

    #[must_use]
    pub fn path(&self) -> Option<&str> {
        self.path.as_deref()
    }

    fn new(code: SourceErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            path: None,
            message: message.into(),
            source: None,
        }
    }

    fn at_path(code: SourceErrorCode, path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code,
            path: Some(path.into()),
            message: message.into(),
            source: None,
        }
    }

    fn io(error: std::io::Error, message: impl Into<String>) -> Self {
        Self {
            code: SourceErrorCode::Io,
            path: None,
            message: message.into(),
            source: Some(Box::new(error)),
        }
    }

    fn path_io(error: std::io::Error, path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: SourceErrorCode::Io,
            path: Some(path.into()),
            message: message.into(),
            source: Some(Box::new(error)),
        }
    }
}

impl fmt::Display for SourceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "compiler source error {:?}: {}",
            self.code, self.message
        )?;
        if let Some(path) = &self.path {
            write!(formatter, " (path {path:?})")?;
        }
        Ok(())
    }
}

impl Error for SourceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn Error + 'static))
    }
}

impl From<pvlc_pack::PackError> for SourceError {
    fn from(error: pvlc_pack::PackError) -> Self {
        Self {
            code: SourceErrorCode::Pack,
            path: None,
            message: error.to_string(),
            source: Some(Box::new(error)),
        }
    }
}

impl From<pvlc_ir::GraphError> for SourceError {
    fn from(error: pvlc_ir::GraphError) -> Self {
        Self {
            code: SourceErrorCode::Pack,
            path: None,
            message: error.to_string(),
            source: Some(Box::new(error)),
        }
    }
}
