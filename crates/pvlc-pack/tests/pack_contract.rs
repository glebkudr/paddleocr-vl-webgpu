use pvlc_model_schema::{COMPILER_MODEL_ABI, MODEL_ID, MODEL_REVISION};
use pvlc_pack::{
    PACK_FORMAT_VERSION, PACK_MAGIC, PackBuilder, PackErrorCode, PackManifest, PackReader,
    PackSection, PrecisionProfile, SectionKind,
};

const EXPECTED_TINY_PACK_LEN: usize = 841;
const EXPECTED_TINY_PACK_BLAKE3: &str =
    "01fe6f47b810877d4c2bb17bc68d4c4aef1fee4d21fc7216837d4005251346c6";
const HEADER_BYTES: usize = 32;
const DIRECTORY_FIXED_BYTES: usize = 56;

fn manifest() -> PackManifest {
    PackManifest {
        model_id: MODEL_ID.to_owned(),
        model_revision: MODEL_REVISION.to_owned(),
        compiler_model_abi: COMPILER_MODEL_ABI,
        compiler_build: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
        precision_profile: PrecisionProfile::Fidelity,
        resolution_buckets: vec![[28, 28], [56, 84]],
        context_limit: 128,
    }
}

fn sections() -> Vec<PackSection> {
    vec![
        PackSection::new(
            "weights.tiny",
            SectionKind::WeightShard,
            256,
            (0..73).map(|i| (i * 17 + 3) as u8).collect(),
        ),
        PackSection::new(
            "self_test.tiny",
            SectionKind::SelfTest,
            16,
            vec![0xff, 0x00, 0x7f, 0x80, 0x01],
        ),
        PackSection::new(
            "ir.semantic",
            SectionKind::SemanticIr,
            64,
            b"{\"nodes\":[]}\n".to_vec(),
        ),
    ]
}

fn build_with_order(order: &[usize]) -> Vec<u8> {
    let mut builder = PackBuilder::new(manifest());
    let sections = sections();
    for index in order {
        builder.add_section(sections[*index].clone()).unwrap();
    }
    builder.build().unwrap()
}

fn align_up(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

fn canonical_manifest_bytes() -> Vec<u8> {
    let value = serde_json::json!({
        "compiler_build": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "compiler_model_abi": 1,
        "context_limit": 128,
        "model_id": "PaddlePaddle/PaddleOCR-VL-1.6",
        "model_revision": "66317acc4c9fc17bd154591ce650735cd2855f3e",
        "precision_profile": "fidelity",
        "resolution_buckets": [[28, 28], [56, 84]],
    });
    let mut bytes = serde_json::to_vec(&value).unwrap();
    bytes.push(b'\n');
    bytes
}

#[derive(Clone)]
struct ExpectedSection {
    id: String,
    tag: u8,
    alignment: usize,
    bytes: Vec<u8>,
    offset: usize,
}

#[derive(Debug)]
struct DirectoryLocation {
    entry_start: usize,
    id_range: std::ops::Range<usize>,
    padding_range: std::ops::Range<usize>,
    offset: usize,
    byte_len: usize,
}

fn directory_locations(bytes: &[u8]) -> Vec<DirectoryLocation> {
    let manifest_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let section_count = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;
    let mut cursor = HEADER_BYTES + manifest_len;
    let mut locations = Vec::new();
    for _ in 0..section_count {
        let entry_start = cursor;
        let id_len = u16::from_le_bytes(bytes[cursor..cursor + 2].try_into().unwrap()) as usize;
        let offset =
            u64::from_le_bytes(bytes[cursor + 8..cursor + 16].try_into().unwrap()) as usize;
        let byte_len =
            u64::from_le_bytes(bytes[cursor + 16..cursor + 24].try_into().unwrap()) as usize;
        let id_range = cursor + DIRECTORY_FIXED_BYTES..cursor + DIRECTORY_FIXED_BYTES + id_len;
        cursor = entry_start + align_up(DIRECTORY_FIXED_BYTES + id_len, 8);
        locations.push(DirectoryLocation {
            entry_start,
            padding_range: id_range.end..cursor,
            id_range,
            offset,
            byte_len,
        });
    }
    locations
}

/// Independent binary encoder for the format contract. It intentionally does
/// not use `PackBuilder`, manifest serialization, section tags, or align helpers
/// from production.
fn independently_encoded_tiny_pack() -> Vec<u8> {
    let manifest = canonical_manifest_bytes();
    let mut sections = vec![
        ExpectedSection {
            id: "weights.tiny".into(),
            tag: 2,
            alignment: 256,
            bytes: (0..73).map(|i| (i * 17 + 3) as u8).collect(),
            offset: 0,
        },
        ExpectedSection {
            id: "self_test.tiny".into(),
            tag: 3,
            alignment: 16,
            bytes: vec![0xff, 0x00, 0x7f, 0x80, 0x01],
            offset: 0,
        },
        ExpectedSection {
            id: "ir.semantic".into(),
            tag: 1,
            alignment: 64,
            bytes: b"{\"nodes\":[]}\n".to_vec(),
            offset: 0,
        },
    ];
    sections.sort_by(|a, b| a.id.cmp(&b.id));
    let directory_len: usize = sections
        .iter()
        .map(|section| align_up(DIRECTORY_FIXED_BYTES + section.id.len(), 8))
        .sum();
    let mut cursor = HEADER_BYTES + manifest.len() + directory_len;
    for section in &mut sections {
        cursor = align_up(cursor, section.alignment);
        section.offset = cursor;
        cursor += section.bytes.len();
    }
    let file_len = cursor;

    let mut directory = Vec::with_capacity(directory_len);
    for section in &sections {
        directory.extend_from_slice(&(section.id.len() as u16).to_le_bytes());
        directory.push(section.tag);
        directory.push(0);
        directory.extend_from_slice(&(section.alignment as u32).to_le_bytes());
        directory.extend_from_slice(&(section.offset as u64).to_le_bytes());
        directory.extend_from_slice(&(section.bytes.len() as u64).to_le_bytes());
        directory.extend_from_slice(blake3::hash(&section.bytes).as_bytes());
        directory.extend_from_slice(section.id.as_bytes());
        directory.resize(align_up(directory.len(), 8), 0);
    }
    assert_eq!(directory.len(), directory_len);

    let mut output = Vec::with_capacity(file_len);
    output.extend_from_slice(b"PVLCPK01");
    output.extend_from_slice(&1_u32.to_le_bytes());
    output.extend_from_slice(&(manifest.len() as u32).to_le_bytes());
    output.extend_from_slice(&(directory_len as u32).to_le_bytes());
    output.extend_from_slice(&(sections.len() as u32).to_le_bytes());
    output.extend_from_slice(&(file_len as u64).to_le_bytes());
    output.extend_from_slice(&manifest);
    output.extend_from_slice(&directory);
    for section in sections {
        output.resize(section.offset, 0);
        output.extend_from_slice(&section.bytes);
    }
    assert_eq!(output.len(), file_len);
    output
}

#[test]
fn constants_and_manifest_identity_are_a_stable_abi() {
    assert_eq!(PACK_MAGIC, *b"PVLCPK01");
    assert_eq!(PACK_FORMAT_VERSION, 1);
    let manifest = PackManifest::paddleocr_vl_16(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    .unwrap();
    assert_eq!(manifest.model_id, MODEL_ID);
    assert_eq!(manifest.model_revision, MODEL_REVISION);
    assert_eq!(manifest.compiler_model_abi, COMPILER_MODEL_ABI);
    assert_eq!(manifest.precision_profile, PrecisionProfile::Fidelity);

    for invalid in [
        "a".repeat(63),
        "a".repeat(65),
        "A".repeat(64),
        "g".repeat(64),
    ] {
        assert_eq!(
            PackManifest::paddleocr_vl_16(&invalid).unwrap_err().code(),
            PackErrorCode::InvalidCompilerBuild,
            "compiler build {invalid:?}"
        );
    }
}

#[test]
fn tiny_pack_is_bit_reproducible_and_matches_independent_binary_encoding() {
    let first = build_with_order(&[0, 1, 2]);
    let second = build_with_order(&[2, 0, 1]);
    let expected = independently_encoded_tiny_pack();

    assert_eq!(first, second, "insertion order must not affect output");
    assert_eq!(first, expected);
    assert_eq!(first.len(), EXPECTED_TINY_PACK_LEN);
    assert_eq!(
        blake3::hash(&first).to_hex().as_str(),
        EXPECTED_TINY_PACK_BLAKE3
    );
}

#[test]
fn pack_roundtrip_preserves_manifest_directory_and_payloads() {
    let bytes = build_with_order(&[1, 2, 0]);
    let pack = PackReader::open(&bytes).unwrap();

    assert_eq!(pack.manifest(), &manifest());
    assert_eq!(
        pack.entries()
            .iter()
            .map(|entry| entry.id.as_str())
            .collect::<Vec<_>>(),
        ["ir.semantic", "self_test.tiny", "weights.tiny"]
    );
    assert_eq!(pack.section("ir.semantic").unwrap(), b"{\"nodes\":[]}\n");
    assert_eq!(
        pack.section("self_test.tiny").unwrap(),
        [0xff, 0x00, 0x7f, 0x80, 0x01]
    );
    assert_eq!(pack.section("weights.tiny").unwrap(), sections()[0].bytes);
    assert_eq!(pack.entries()[0].kind, SectionKind::SemanticIr);
    assert_eq!(pack.entries()[2].alignment, 256);
    assert_eq!(pack.entries()[2].digest.len(), 32);
    assert!(pack.section("missing").is_none());
}

#[test]
fn builder_rejects_wrong_model_identity_duplicate_ids_and_bad_alignment() {
    for mutate in [0, 1, 2] {
        let mut wrong = manifest();
        match mutate {
            0 => wrong.model_id = "other/model".into(),
            1 => wrong.model_revision = "0".repeat(40),
            2 => wrong.compiler_model_abi += 1,
            _ => unreachable!(),
        }
        assert_eq!(
            PackBuilder::new(wrong).build().unwrap_err().code(),
            PackErrorCode::ModelIdentityMismatch
        );
    }

    let mut duplicate = PackBuilder::new(manifest());
    duplicate.add_section(sections()[0].clone()).unwrap();
    let error = duplicate.add_section(sections()[0].clone()).unwrap_err();
    assert_eq!(error.code(), PackErrorCode::DuplicateSection);

    for alignment in [0, 3, 8_192] {
        let mut builder = PackBuilder::new(manifest());
        let error = builder
            .add_section(PackSection::new(
                "weights.bad",
                SectionKind::WeightShard,
                alignment,
                vec![1],
            ))
            .unwrap_err();
        assert_eq!(error.code(), PackErrorCode::InvalidAlignment);
    }
    for alignment in [1, 4_096] {
        let mut builder = PackBuilder::new(manifest());
        builder
            .add_section(PackSection::new(
                format!("weights.valid_{alignment}"),
                SectionKind::WeightShard,
                alignment,
                vec![1],
            ))
            .unwrap();
        builder.build().unwrap();
    }
}

#[test]
fn reader_rejects_magic_version_truncation_checksum_and_nonzero_padding() {
    let pristine = build_with_order(&[0, 1, 2]);

    let mut bad_magic = pristine.clone();
    bad_magic[0] ^= 1;
    assert_eq!(
        PackReader::open(&bad_magic).unwrap_err().code(),
        PackErrorCode::BadMagic
    );

    let mut bad_version = pristine.clone();
    bad_version[8..12].copy_from_slice(&2_u32.to_le_bytes());
    assert_eq!(
        PackReader::open(&bad_version).unwrap_err().code(),
        PackErrorCode::UnsupportedVersion
    );

    for length in 0..HEADER_BYTES {
        assert_eq!(
            PackReader::open(&pristine[..length]).unwrap_err().code(),
            PackErrorCode::Truncated,
            "short length {length}"
        );
    }
    assert_eq!(
        PackReader::open(&pristine[..pristine.len() - 1])
            .unwrap_err()
            .code(),
        PackErrorCode::LengthMismatch
    );
    let mut trailing = pristine.clone();
    trailing.push(0);
    assert_eq!(
        PackReader::open(&trailing).unwrap_err().code(),
        PackErrorCode::LengthMismatch
    );
    let mut declared_shorter = pristine.clone();
    declared_shorter[24..32].copy_from_slice(&((pristine.len() - 1) as u64).to_le_bytes());
    assert_eq!(
        PackReader::open(&declared_shorter).unwrap_err().code(),
        PackErrorCode::LengthMismatch
    );

    let locations = directory_locations(&pristine);
    for location in &locations {
        let mut corrupt_payload = pristine.clone();
        corrupt_payload[location.offset + location.byte_len / 2] ^= 1;
        assert_eq!(
            PackReader::open(&corrupt_payload).unwrap_err().code(),
            PackErrorCode::ChecksumMismatch,
            "payload at {}",
            location.offset
        );
    }

    for location in &locations {
        assert!(!location.padding_range.is_empty());
        let mut corrupt_padding = pristine.clone();
        corrupt_padding[location.padding_range.start] = 1;
        assert_eq!(
            PackReader::open(&corrupt_padding).unwrap_err().code(),
            PackErrorCode::NonZeroPadding,
            "directory padding at {}",
            location.padding_range.start
        );
    }

    let manifest_len = u32::from_le_bytes(pristine[12..16].try_into().unwrap()) as usize;
    let directory_len = u32::from_le_bytes(pristine[16..20].try_into().unwrap()) as usize;
    let mut previous_end = HEADER_BYTES + manifest_len + directory_len;
    for location in &locations {
        assert!(previous_end < location.offset);
        let mut corrupt_padding = pristine.clone();
        corrupt_padding[previous_end] = 1;
        assert_eq!(
            PackReader::open(&corrupt_padding).unwrap_err().code(),
            PackErrorCode::NonZeroPadding,
            "section padding at {previous_end}"
        );
        previous_end = location.offset + location.byte_len;
    }
}

#[test]
fn reader_rejects_noncanonical_or_malformed_directory_metadata() {
    let pristine = build_with_order(&[0, 1, 2]);
    let manifest_len = u32::from_le_bytes(pristine[12..16].try_into().unwrap()) as usize;
    let first_entry = HEADER_BYTES + manifest_len;

    let mut bad_reserved = pristine.clone();
    bad_reserved[first_entry + 3] = 1;
    assert_eq!(
        PackReader::open(&bad_reserved).unwrap_err().code(),
        PackErrorCode::InvalidDirectory
    );

    let mut bad_alignment = pristine.clone();
    bad_alignment[first_entry + 4..first_entry + 8].copy_from_slice(&3_u32.to_le_bytes());
    assert_eq!(
        PackReader::open(&bad_alignment).unwrap_err().code(),
        PackErrorCode::InvalidAlignment
    );

    let mut unaligned_offset = pristine;
    let original = u64::from_le_bytes(
        unaligned_offset[first_entry + 8..first_entry + 16]
            .try_into()
            .unwrap(),
    );
    unaligned_offset[first_entry + 8..first_entry + 16]
        .copy_from_slice(&(original + 1).to_le_bytes());
    assert_eq!(
        PackReader::open(&unaligned_offset).unwrap_err().code(),
        PackErrorCode::InvalidSectionLayout
    );

    let mut unknown_tag = build_with_order(&[0, 1, 2]);
    unknown_tag[first_entry + 2] = 255;
    assert_eq!(
        PackReader::open(&unknown_tag).unwrap_err().code(),
        PackErrorCode::UnknownSectionKind
    );

    let mut invalid_utf8 = build_with_order(&[0, 1, 2]);
    let first_id_start = directory_locations(&invalid_utf8)[0].id_range.start;
    invalid_utf8[first_id_start] = 0xff;
    assert_eq!(
        PackReader::open(&invalid_utf8).unwrap_err().code(),
        PackErrorCode::InvalidDirectory
    );

    for count in [2_u32, 4_u32] {
        let mut inconsistent_count = build_with_order(&[0, 1, 2]);
        inconsistent_count[20..24].copy_from_slice(&count.to_le_bytes());
        assert_eq!(
            PackReader::open(&inconsistent_count).unwrap_err().code(),
            PackErrorCode::InvalidDirectory,
            "section count {count}"
        );
    }

    let mut noncanonical = build_with_order(&[0, 1, 2]);
    let first = &directory_locations(&noncanonical)[0];
    assert_eq!(first.id_range.len(), b"zz.semantic".len());
    noncanonical[first.id_range.clone()].copy_from_slice(b"zz.semantic");
    assert_eq!(
        PackReader::open(&noncanonical).unwrap_err().code(),
        PackErrorCode::NonCanonicalDirectory
    );
}

#[test]
fn reader_never_panics_on_hostile_lengths_counts_offsets_or_overlaps() {
    let pristine = build_with_order(&[0, 1, 2]);

    for range in [12..16, 16..20, 20..24] {
        let mut hostile = pristine.clone();
        hostile[range].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            PackReader::open(&hostile).unwrap_err().code(),
            PackErrorCode::InvalidHeader
        );
    }
    let mut impossible_file_len = pristine.clone();
    impossible_file_len[24..32].copy_from_slice(&u64::MAX.to_le_bytes());
    assert_eq!(
        PackReader::open(&impossible_file_len).unwrap_err().code(),
        PackErrorCode::LengthMismatch
    );

    let locations = directory_locations(&pristine);
    let first = locations[0].entry_start;
    let second = locations[1].entry_start;

    let mut huge_id = pristine.clone();
    huge_id[first..first + 2].copy_from_slice(&u16::MAX.to_le_bytes());
    assert_eq!(
        PackReader::open(&huge_id).unwrap_err().code(),
        PackErrorCode::InvalidDirectory
    );

    for field in [first + 8..first + 16, first + 16..first + 24] {
        let mut huge = pristine.clone();
        huge[field].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(
            PackReader::open(&huge).unwrap_err().code(),
            PackErrorCode::InvalidSectionLayout
        );
    }

    let mut overlap = pristine.clone();
    overlap[second + 8..second + 16].copy_from_slice(&(locations[0].offset as u64).to_le_bytes());
    assert_eq!(
        PackReader::open(&overlap).unwrap_err().code(),
        PackErrorCode::InvalidSectionLayout
    );

    let mut out_of_bounds = pristine;
    out_of_bounds[second + 16..second + 24].copy_from_slice(&u64::MAX.to_le_bytes());
    assert_eq!(
        PackReader::open(&out_of_bounds).unwrap_err().code(),
        PackErrorCode::InvalidSectionLayout
    );
}

#[test]
fn reader_rejects_duplicate_directory_ids_even_when_lengths_match() {
    let mut builder = PackBuilder::new(manifest());
    builder
        .add_section(PackSection::new(
            "test.aaaa",
            SectionKind::SelfTest,
            8,
            vec![1],
        ))
        .unwrap();
    builder
        .add_section(PackSection::new(
            "test.bbbb",
            SectionKind::SelfTest,
            8,
            vec![2],
        ))
        .unwrap();
    let mut bytes = builder.build().unwrap();
    let locations = directory_locations(&bytes);
    let first_id = bytes[locations[0].id_range.clone()].to_vec();
    bytes[locations[1].id_range.clone()].copy_from_slice(&first_id);
    assert_eq!(
        PackReader::open(&bytes).unwrap_err().code(),
        PackErrorCode::DuplicateSection
    );
}
