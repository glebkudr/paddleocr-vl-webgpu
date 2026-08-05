//! Deterministic, checksummed PVLC model-pack format.

mod projector_self_test;
mod vision_layer_self_test;
mod vision_stack_shards;

pub use projector_self_test::*;
pub use vision_layer_self_test::*;
pub use vision_stack_shards::*;

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use pvlc_ir::SemanticId;
use pvlc_model_schema::{COMPILER_MODEL_ABI, MODEL_ID, MODEL_REVISION};
use serde::{Deserialize, Serialize};

pub const PACK_MAGIC: [u8; 8] = *b"PVLCPK01";
pub const PACK_FORMAT_VERSION: u32 = 1;

const HEADER_BYTES: usize = 32;
const DIRECTORY_FIXED_BYTES: usize = 56;
const MAX_MANIFEST_BYTES: usize = 1_048_576;
const MAX_DIRECTORY_BYTES: usize = 16_777_216;
const MAX_SECTIONS: usize = 65_536;
const MAX_ALIGNMENT: u32 = 4_096;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PrecisionProfile {
    Fidelity,
    Balanced,
    Turbo,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackManifest {
    pub model_id: String,
    pub model_revision: String,
    pub compiler_model_abi: u32,
    pub compiler_build: String,
    pub precision_profile: PrecisionProfile,
    pub resolution_buckets: Vec<[u32; 2]>,
    pub context_limit: u32,
}

impl PackManifest {
    pub fn paddleocr_vl_16(compiler_build: impl Into<String>) -> Result<Self, PackError> {
        let manifest = Self {
            model_id: MODEL_ID.to_owned(),
            model_revision: MODEL_REVISION.to_owned(),
            compiler_model_abi: COMPILER_MODEL_ABI,
            compiler_build: compiler_build.into(),
            precision_profile: PrecisionProfile::Fidelity,
            resolution_buckets: vec![[28, 28]],
            context_limit: 4_096,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    fn validate(&self) -> Result<(), PackError> {
        if self.model_id != MODEL_ID
            || self.model_revision != MODEL_REVISION
            || self.compiler_model_abi != COMPILER_MODEL_ABI
        {
            return Err(PackError::new(
                PackErrorCode::ModelIdentityMismatch,
                "pack manifest does not identify the pinned PaddleOCR-VL model",
            ));
        }
        if !is_lower_hex_64(&self.compiler_build) {
            return Err(PackError::new(
                PackErrorCode::InvalidCompilerBuild,
                "compiler build must be exactly 64 lowercase hexadecimal characters",
            ));
        }
        if self.context_limit == 0
            || self.resolution_buckets.is_empty()
            || self
                .resolution_buckets
                .iter()
                .any(|bucket| bucket[0] == 0 || bucket[1] == 0)
        {
            return Err(PackError::new(
                PackErrorCode::InvalidManifest,
                "context limit and resolution buckets must be nonzero",
            ));
        }
        Ok(())
    }

    fn canonical_bytes(&self) -> Result<Vec<u8>, PackError> {
        self.validate()?;
        #[derive(Serialize)]
        struct CanonicalManifest<'a> {
            compiler_build: &'a str,
            compiler_model_abi: u32,
            context_limit: u32,
            model_id: &'a str,
            model_revision: &'a str,
            precision_profile: PrecisionProfile,
            resolution_buckets: &'a [[u32; 2]],
        }
        let canonical = CanonicalManifest {
            compiler_build: &self.compiler_build,
            compiler_model_abi: self.compiler_model_abi,
            context_limit: self.context_limit,
            model_id: &self.model_id,
            model_revision: &self.model_revision,
            precision_profile: self.precision_profile,
            resolution_buckets: &self.resolution_buckets,
        };
        let mut bytes = serde_json::to_vec(&canonical).map_err(|error| {
            PackError::new(
                PackErrorCode::InvalidManifest,
                format!("cannot serialize manifest: {error}"),
            )
        })?;
        bytes.push(b'\n');
        Ok(bytes)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SectionKind {
    SemanticIr,
    WeightShard,
    SelfTest,
    ModelSchema,
    SemanticMap,
}

impl SectionKind {
    const fn tag(self) -> u8 {
        match self {
            Self::SemanticIr => 1,
            Self::WeightShard => 2,
            Self::SelfTest => 3,
            Self::ModelSchema => 4,
            Self::SemanticMap => 5,
        }
    }

    fn from_tag(tag: u8) -> Result<Self, PackError> {
        match tag {
            1 => Ok(Self::SemanticIr),
            2 => Ok(Self::WeightShard),
            3 => Ok(Self::SelfTest),
            4 => Ok(Self::ModelSchema),
            5 => Ok(Self::SemanticMap),
            _ => Err(PackError::new(
                PackErrorCode::UnknownSectionKind,
                format!("unknown section kind tag {tag}"),
            )),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackSection {
    pub id: String,
    pub kind: SectionKind,
    pub alignment: u32,
    pub bytes: Vec<u8>,
}

impl PackSection {
    #[must_use]
    pub fn new(id: impl Into<String>, kind: SectionKind, alignment: u32, bytes: Vec<u8>) -> Self {
        Self {
            id: id.into(),
            kind,
            alignment,
            bytes,
        }
    }
}

#[derive(Debug)]
pub struct PackBuilder {
    manifest: PackManifest,
    sections: BTreeMap<String, PackSection>,
}

impl PackBuilder {
    #[must_use]
    pub fn new(manifest: PackManifest) -> Self {
        Self {
            manifest,
            sections: BTreeMap::new(),
        }
    }

    pub fn add_section(&mut self, section: PackSection) -> Result<(), PackError> {
        validate_section_id(&section.id)?;
        validate_alignment(section.alignment)?;
        if self.sections.contains_key(&section.id) {
            return Err(PackError::new(
                PackErrorCode::DuplicateSection,
                format!("section {:?} is duplicated", section.id),
            ));
        }
        self.sections.insert(section.id.clone(), section);
        Ok(())
    }

    pub fn build(self) -> Result<Vec<u8>, PackError> {
        let manifest_bytes = self.manifest.canonical_bytes()?;
        let manifest_len = u32::try_from(manifest_bytes.len())
            .map_err(|_| PackError::new(PackErrorCode::InvalidManifest, "manifest exceeds u32"))?;
        let section_count = u32::try_from(self.sections.len()).map_err(|_| {
            PackError::new(PackErrorCode::InvalidDirectory, "too many pack sections")
        })?;

        let mut directory_len = 0_usize;
        for section in self.sections.values() {
            let unpadded = DIRECTORY_FIXED_BYTES
                .checked_add(section.id.len())
                .ok_or_else(|| overflow_error("directory entry"))?;
            let entry_len = align_up(unpadded, 8)?;
            directory_len = directory_len
                .checked_add(entry_len)
                .ok_or_else(|| overflow_error("directory"))?;
        }
        let directory_len_u32 = u32::try_from(directory_len).map_err(|_| {
            PackError::new(PackErrorCode::InvalidDirectory, "directory exceeds u32")
        })?;

        let mut cursor = HEADER_BYTES
            .checked_add(manifest_bytes.len())
            .and_then(|value| value.checked_add(directory_len))
            .ok_or_else(|| overflow_error("pack prefix"))?;
        let mut entries = Vec::with_capacity(self.sections.len());
        for section in self.sections.values() {
            cursor = align_up(cursor, section.alignment as usize)?;
            let offset = cursor;
            cursor = cursor
                .checked_add(section.bytes.len())
                .ok_or_else(|| overflow_error("pack payload"))?;
            entries.push(BuildEntry {
                section,
                offset,
                digest: *blake3::hash(&section.bytes).as_bytes(),
            });
        }
        let file_len = u64::try_from(cursor)
            .map_err(|_| PackError::new(PackErrorCode::InvalidHeader, "pack exceeds u64"))?;

        let mut directory = Vec::with_capacity(directory_len);
        for entry in &entries {
            let section = entry.section;
            directory.extend_from_slice(&(section.id.len() as u16).to_le_bytes());
            directory.push(section.kind.tag());
            directory.push(0);
            directory.extend_from_slice(&section.alignment.to_le_bytes());
            directory.extend_from_slice(&(entry.offset as u64).to_le_bytes());
            directory.extend_from_slice(&(section.bytes.len() as u64).to_le_bytes());
            directory.extend_from_slice(&entry.digest);
            directory.extend_from_slice(section.id.as_bytes());
            let padded = align_up(directory.len(), 8)?;
            directory.resize(padded, 0);
        }
        debug_assert_eq!(directory.len(), directory_len);

        let mut output = Vec::with_capacity(cursor);
        output.extend_from_slice(&PACK_MAGIC);
        output.extend_from_slice(&PACK_FORMAT_VERSION.to_le_bytes());
        output.extend_from_slice(&manifest_len.to_le_bytes());
        output.extend_from_slice(&directory_len_u32.to_le_bytes());
        output.extend_from_slice(&section_count.to_le_bytes());
        output.extend_from_slice(&file_len.to_le_bytes());
        output.extend_from_slice(&manifest_bytes);
        output.extend_from_slice(&directory);
        for entry in entries {
            output.resize(entry.offset, 0);
            output.extend_from_slice(&entry.section.bytes);
        }
        debug_assert_eq!(output.len(), cursor);
        Ok(output)
    }
}

struct BuildEntry<'a> {
    section: &'a PackSection,
    offset: usize,
    digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackEntry {
    pub id: String,
    pub kind: SectionKind,
    pub alignment: u32,
    pub offset: u64,
    pub byte_len: u64,
    pub digest: [u8; 32],
}

#[derive(Debug)]
pub struct PackReader<'a> {
    bytes: &'a [u8],
    manifest: PackManifest,
    entries: Vec<PackEntry>,
}

impl<'a> PackReader<'a> {
    pub fn open(bytes: &'a [u8]) -> Result<Self, PackError> {
        if bytes.len() < HEADER_BYTES {
            return Err(PackError::new(
                PackErrorCode::Truncated,
                "pack is shorter than its fixed header",
            ));
        }
        if bytes[..8] != PACK_MAGIC {
            return Err(PackError::new(
                PackErrorCode::BadMagic,
                "pack magic mismatch",
            ));
        }
        let version = read_u32(bytes, 8);
        if version != PACK_FORMAT_VERSION {
            return Err(PackError::new(
                PackErrorCode::UnsupportedVersion,
                format!("unsupported pack version {version}"),
            ));
        }
        let manifest_len = read_u32(bytes, 12) as usize;
        let directory_len = read_u32(bytes, 16) as usize;
        let section_count = read_u32(bytes, 20) as usize;
        let declared_file_len = read_u64(bytes, 24);
        if declared_file_len != bytes.len() as u64 {
            return Err(PackError::new(
                PackErrorCode::LengthMismatch,
                format!(
                    "header declares {declared_file_len} bytes but input has {}",
                    bytes.len()
                ),
            ));
        }
        if manifest_len > MAX_MANIFEST_BYTES
            || directory_len > MAX_DIRECTORY_BYTES
            || section_count > MAX_SECTIONS
        {
            return Err(PackError::new(
                PackErrorCode::InvalidHeader,
                "manifest, directory, or section count exceeds its bound",
            ));
        }
        let manifest_start = HEADER_BYTES;
        let manifest_end = manifest_start
            .checked_add(manifest_len)
            .ok_or_else(|| overflow_error("manifest bounds"))?;
        let directory_end = manifest_end
            .checked_add(directory_len)
            .ok_or_else(|| overflow_error("directory bounds"))?;
        if directory_end > bytes.len() {
            return Err(PackError::new(
                PackErrorCode::InvalidHeader,
                "manifest and directory extend beyond the pack",
            ));
        }
        let manifest = parse_canonical_manifest(&bytes[manifest_start..manifest_end])?;

        let mut entries = Vec::with_capacity(section_count);
        let mut cursor = manifest_end;
        let mut previous_id: Option<String> = None;
        for _ in 0..section_count {
            let fixed_end = cursor
                .checked_add(DIRECTORY_FIXED_BYTES)
                .ok_or_else(|| overflow_error("directory entry"))?;
            if fixed_end > directory_end {
                return Err(PackError::new(
                    PackErrorCode::InvalidDirectory,
                    "directory entry fixed fields are truncated",
                ));
            }
            let id_len = read_u16(bytes, cursor) as usize;
            let tag = bytes[cursor + 2];
            let reserved = bytes[cursor + 3];
            let alignment = read_u32(bytes, cursor + 4);
            let offset = read_u64(bytes, cursor + 8);
            let byte_len = read_u64(bytes, cursor + 16);
            let mut digest = [0_u8; 32];
            digest.copy_from_slice(&bytes[cursor + 24..cursor + 56]);
            if reserved != 0 {
                return Err(PackError::new(
                    PackErrorCode::InvalidDirectory,
                    "directory reserved byte is nonzero",
                ));
            }
            let kind = SectionKind::from_tag(tag)?;
            validate_alignment(alignment)?;
            if id_len == 0 || id_len > 128 {
                return Err(PackError::new(
                    PackErrorCode::InvalidDirectory,
                    "section ID length is out of bounds",
                ));
            }
            let id_end = fixed_end
                .checked_add(id_len)
                .ok_or_else(|| overflow_error("section ID"))?;
            let entry_len = align_up(DIRECTORY_FIXED_BYTES + id_len, 8)?;
            let entry_end = cursor
                .checked_add(entry_len)
                .ok_or_else(|| overflow_error("directory entry padding"))?;
            if entry_end > directory_end {
                return Err(PackError::new(
                    PackErrorCode::InvalidDirectory,
                    "section ID or entry padding is truncated",
                ));
            }
            let id = std::str::from_utf8(&bytes[fixed_end..id_end]).map_err(|_| {
                PackError::new(
                    PackErrorCode::InvalidDirectory,
                    "section ID is not valid UTF-8",
                )
            })?;
            validate_section_id(id).map_err(|_| {
                PackError::new(
                    PackErrorCode::InvalidDirectory,
                    format!("invalid section ID {id:?}"),
                )
            })?;
            require_zero(&bytes[id_end..entry_end])?;
            if let Some(previous) = &previous_id {
                match id.cmp(previous) {
                    std::cmp::Ordering::Equal => {
                        return Err(PackError::new(
                            PackErrorCode::DuplicateSection,
                            format!("directory repeats section {id:?}"),
                        ));
                    }
                    std::cmp::Ordering::Less => {
                        return Err(PackError::new(
                            PackErrorCode::NonCanonicalDirectory,
                            "section IDs are not in canonical byte order",
                        ));
                    }
                    std::cmp::Ordering::Greater => {}
                }
            }
            previous_id = Some(id.to_owned());
            entries.push(PackEntry {
                id: id.to_owned(),
                kind,
                alignment,
                offset,
                byte_len,
                digest,
            });
            cursor = entry_end;
        }
        if cursor != directory_end {
            return Err(PackError::new(
                PackErrorCode::InvalidDirectory,
                "section count does not consume the complete directory",
            ));
        }

        let mut previous_end = directory_end;
        for entry in &entries {
            let expected_offset = align_up(previous_end, entry.alignment as usize)?;
            let offset = usize::try_from(entry.offset).map_err(|_| layout_error())?;
            let byte_len = usize::try_from(entry.byte_len).map_err(|_| layout_error())?;
            let end = offset.checked_add(byte_len).ok_or_else(layout_error)?;
            if offset != expected_offset || end > bytes.len() {
                return Err(layout_error());
            }
            require_zero(&bytes[previous_end..offset])?;
            previous_end = end;
        }
        if previous_end != bytes.len() {
            return Err(layout_error());
        }
        for entry in &entries {
            let start = entry.offset as usize;
            let end = start + entry.byte_len as usize;
            if blake3::hash(&bytes[start..end]).as_bytes() != &entry.digest {
                return Err(PackError::new(
                    PackErrorCode::ChecksumMismatch,
                    format!("section {:?} checksum mismatch", entry.id),
                ));
            }
        }

        Ok(Self {
            bytes,
            manifest,
            entries,
        })
    }

    #[must_use]
    pub const fn manifest(&self) -> &PackManifest {
        &self.manifest
    }

    #[must_use]
    pub fn entries(&self) -> &[PackEntry] {
        &self.entries
    }

    #[must_use]
    pub fn section(&self, id: &str) -> Option<&'a [u8]> {
        let index = self
            .entries
            .binary_search_by(|entry| entry.id.as_str().cmp(id))
            .ok()?;
        let entry = &self.entries[index];
        let start = entry.offset as usize;
        let end = start + entry.byte_len as usize;
        Some(&self.bytes[start..end])
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackErrorCode {
    ModelIdentityMismatch,
    InvalidCompilerBuild,
    InvalidManifest,
    InvalidSectionId,
    DuplicateSection,
    InvalidAlignment,
    BadMagic,
    UnsupportedVersion,
    Truncated,
    LengthMismatch,
    InvalidHeader,
    InvalidDirectory,
    UnknownSectionKind,
    NonCanonicalDirectory,
    InvalidSectionLayout,
    ChecksumMismatch,
    NonZeroPadding,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackError {
    code: PackErrorCode,
    message: String,
}

impl PackError {
    #[must_use]
    pub const fn code(&self) -> PackErrorCode {
        self.code
    }

    fn new(code: PackErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl fmt::Display for PackError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "pack error {:?}: {}", self.code, self.message)
    }
}

impl Error for PackError {}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OwnedCanonicalManifest {
    compiler_build: String,
    compiler_model_abi: u32,
    context_limit: u32,
    model_id: String,
    model_revision: String,
    precision_profile: PrecisionProfile,
    resolution_buckets: Vec<[u32; 2]>,
}

fn parse_canonical_manifest(bytes: &[u8]) -> Result<PackManifest, PackError> {
    if bytes.last() != Some(&b'\n') {
        return Err(PackError::new(
            PackErrorCode::InvalidManifest,
            "canonical manifest must end in one newline",
        ));
    }
    let text = std::str::from_utf8(bytes)
        .map_err(|_| PackError::new(PackErrorCode::InvalidManifest, "manifest is not UTF-8"))?;
    let owned: OwnedCanonicalManifest = serde_json::from_str(text).map_err(|error| {
        PackError::new(
            PackErrorCode::InvalidManifest,
            format!("invalid manifest JSON: {error}"),
        )
    })?;
    let manifest = PackManifest {
        model_id: owned.model_id,
        model_revision: owned.model_revision,
        compiler_model_abi: owned.compiler_model_abi,
        compiler_build: owned.compiler_build,
        precision_profile: owned.precision_profile,
        resolution_buckets: owned.resolution_buckets,
        context_limit: owned.context_limit,
    };
    if manifest.canonical_bytes()? != bytes {
        return Err(PackError::new(
            PackErrorCode::InvalidManifest,
            "manifest JSON is not in canonical byte form",
        ));
    }
    Ok(manifest)
}

fn validate_section_id(id: &str) -> Result<(), PackError> {
    if SemanticId::parse(id).is_err() && !valid_artifact_section_id(id) {
        return Err(PackError::new(
            PackErrorCode::InvalidSectionId,
            format!("invalid section ID {id:?}"),
        ));
    }
    if id.len() > u16::MAX as usize {
        return Err(PackError::new(
            PackErrorCode::InvalidSectionId,
            "section ID exceeds u16",
        ));
    }
    Ok(())
}

fn valid_artifact_section_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id.split('.').all(|segment| {
            let mut bytes = segment.bytes();
            bytes.next().is_some_and(|first| first.is_ascii_lowercase())
                && bytes.all(|byte| {
                    byte.is_ascii_lowercase()
                        || byte.is_ascii_digit()
                        || matches!(byte, b'_' | b'-')
                })
        })
}

fn validate_alignment(alignment: u32) -> Result<(), PackError> {
    if alignment == 0 || alignment > MAX_ALIGNMENT || !alignment.is_power_of_two() {
        return Err(PackError::new(
            PackErrorCode::InvalidAlignment,
            format!("invalid section alignment {alignment}"),
        ));
    }
    Ok(())
}

fn is_lower_hex_64(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_zero(bytes: &[u8]) -> Result<(), PackError> {
    if bytes.iter().any(|byte| *byte != 0) {
        return Err(PackError::new(
            PackErrorCode::NonZeroPadding,
            "canonical pack padding contains a nonzero byte",
        ));
    }
    Ok(())
}

fn align_up(value: usize, alignment: usize) -> Result<usize, PackError> {
    debug_assert!(alignment != 0 && alignment.is_power_of_two());
    value
        .checked_add(alignment - 1)
        .map(|rounded| rounded & !(alignment - 1))
        .ok_or_else(|| overflow_error("alignment"))
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(
        bytes[offset..offset + 2]
            .try_into()
            .expect("bounds checked"),
    )
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("bounds checked"),
    )
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("bounds checked"),
    )
}

fn overflow_error(context: &str) -> PackError {
    PackError::new(
        PackErrorCode::InvalidHeader,
        format!("integer overflow while computing {context}"),
    )
}

fn layout_error() -> PackError {
    PackError::new(
        PackErrorCode::InvalidSectionLayout,
        "section offsets are noncanonical, overlapping, or out of bounds",
    )
}
