use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use pvlc_model_schema::{MODEL_REVISION, SchemaErrorCode, TensorDtype};
use pvlc_safetensors::{COPY_BUFFER_BYTES, ImportErrorCode, MAX_HEADER_BYTES, SafetensorsCatalog};
use tempfile::TempDir;

#[path = "support/stream_instrumentation.rs"]
mod stream_instrumentation;
use stream_instrumentation::{InstrumentedReader, PartialThenFailWriter};

const EXPECTED_MAX_HEADER_BYTES: u64 = 100_000_000;
const EXPECTED_COPY_BUFFER_BYTES: usize = 64 * 1_024;

fn padded_header(json: &[u8]) -> Vec<u8> {
    let mut header = json.to_vec();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    header
}

fn write_container(path: &Path, header_json: &[u8], body: &[u8]) {
    let bytes = container_bytes(header_json, body);
    let mut file = File::create(path).unwrap();
    file.write_all(&bytes).unwrap();
    file.sync_all().unwrap();
}

fn container_bytes(header_json: &[u8], body: &[u8]) -> Vec<u8> {
    let header = padded_header(header_json);
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(body);
    bytes
}

fn fixture(header_json: &[u8], body: &[u8]) -> (TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fixture.safetensors");
    write_container(&path, header_json, body);
    (dir, path)
}

fn valid_two_tensor_header() -> &'static [u8] {
    // Deliberately reverse lexical and physical order: canonical catalog order
    // must still be `a`, `z`, while offsets remain validated by byte position.
    br#"{"__metadata__":{"format":"pt","producer":"contract-test"},"z":{"dtype":"F32","shape":[1],"data_offsets":[4,8]},"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#
}

#[test]
fn imports_sorted_metadata_and_streams_exact_payload_bytes() {
    let (_dir, path) = fixture(valid_two_tensor_header(), &[0, 1, 2, 3, 4, 5, 6, 7]);
    let catalog = SafetensorsCatalog::open(&path).expect("valid fixture");

    assert_eq!(catalog.file_len(), fs::metadata(&path).unwrap().len());
    assert_eq!(catalog.header_len() % 8, 0);
    assert_eq!(catalog.tensors().len(), 2);
    assert_eq!(catalog.tensors()[0].name, "a");
    assert_eq!(catalog.tensors()[0].dtype, TensorDtype::BFloat16);
    assert_eq!(catalog.tensors()[0].shape, [2]);
    assert_eq!(catalog.tensors()[0].data_offsets, [0, 4]);
    assert_eq!(catalog.tensors()[0].byte_len(), 4);
    assert_eq!(catalog.tensors()[1].name, "z");
    assert_eq!(
        catalog.metadata().get("format").map(String::as_str),
        Some("pt")
    );

    let mut bytes = Vec::new();
    let copied = catalog.copy_tensor_to("z", &mut bytes).unwrap();
    assert_eq!(copied, 4);
    assert_eq!(bytes, [4, 5, 6, 7]);

    let error = catalog
        .copy_tensor_to("missing", &mut Vec::new())
        .unwrap_err();
    assert_eq!(error.code(), ImportErrorCode::TensorNotFound);
    assert_eq!(error.tensor_name(), Some("missing"));
}

#[test]
fn materializes_bfloat16_and_float32_payloads_as_exact_f32_values() {
    let mut body = Vec::new();
    for bits in [0x0000_u16, 0x3f80, 0xbf20, 0x7f7f] {
        body.extend_from_slice(&bits.to_le_bytes());
    }
    for value in [0.5_f32, -2.25, std::f32::consts::PI, -0.0] {
        body.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    body.push(7);
    let (_dir, path) = fixture(
        br#"{"bf":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]},"fp":{"dtype":"F32","shape":[2,2],"data_offsets":[8,24]},"byte":{"dtype":"U8","shape":[1],"data_offsets":[24,25]}}"#,
        &body,
    );
    let catalog = SafetensorsCatalog::open(path).unwrap();

    let bfloat = catalog.load_tensor_f32("bf").unwrap();
    assert_eq!(
        bfloat
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [0x0000_0000, 0x3f80_0000, 0xbf20_0000, 0x7f7f_0000]
    );
    let float = catalog.load_tensor_f32("fp").unwrap();
    assert_eq!(
        float
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [
            0.5_f32.to_bits(),
            (-2.25_f32).to_bits(),
            std::f32::consts::PI.to_bits(),
            (-0.0_f32).to_bits(),
        ]
    );

    let error = catalog.load_tensor_f32("byte").unwrap_err();
    assert_eq!(error.code(), ImportErrorCode::UnsupportedTensorConversion);
    assert_eq!(error.tensor_name(), Some("byte"));
    let error = catalog.load_tensor_f32("missing").unwrap_err();
    assert_eq!(error.code(), ImportErrorCode::TensorNotFound);
    assert_eq!(error.tensor_name(), Some("missing"));
}

#[test]
fn real_pinned_checkpoint_header_matches_the_exact_schema_without_loading_weights() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/snapshots")
        .join(MODEL_REVISION)
        .join("model.safetensors");
    if !path.is_file() {
        assert_ne!(
            std::env::var("PVLC_REQUIRE_MODEL").as_deref(),
            Ok("1"),
            "PVLC_REQUIRE_MODEL=1 but pinned checkpoint is absent at {}",
            path.display()
        );
        eprintln!(
            "skipping local checkpoint assertion: {} is absent",
            path.display()
        );
        return;
    }

    let catalog = SafetensorsCatalog::open(&path).expect("pinned checkpoint header must parse");
    assert_eq!(catalog.file_len(), 1_917_255_968);
    assert_eq!(catalog.header_len(), 78_488);
    assert_eq!(catalog.tensors().len(), 620);
    assert_eq!(
        catalog.metadata().get("format").map(String::as_str),
        Some("pt")
    );
    catalog
        .validate_paddleocr_vl_16()
        .expect("checkpoint must match exact model schema");

    let first = catalog.tensor("lm_head.weight").unwrap();
    assert_eq!(first.data_offsets, [0, 211_812_352]);
    let last = catalog
        .tensor("visual.vision_model.post_layernorm.weight")
        .unwrap();
    assert_eq!(last.data_offsets, [1_917_175_168, 1_917_177_472]);
    assert!(catalog.tensors().windows(2).all(|w| w[0].name < w[1].name));
}

#[test]
fn schema_validation_preserves_precise_model_mismatch_codes() {
    // Use a tiny valid container and prove that model validation—not container
    // parsing—reports the precise semantic mismatch.
    let (_dir, path) = fixture(
        br#"{"lm_head.weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#,
        &[0, 0],
    );
    let catalog = SafetensorsCatalog::open(path).unwrap();
    let error = catalog.validate_paddleocr_vl_16().unwrap_err();
    assert_eq!(error.code(), SchemaErrorCode::ShapeMismatch);
    assert_eq!(error.tensor_name(), Some("lm_head.weight"));
}

#[test]
fn rejects_truncated_oversized_or_unaligned_headers_before_allocation() {
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(MAX_HEADER_BYTES, EXPECTED_MAX_HEADER_BYTES);

    let short_prefix = dir.path().join("short-prefix.safetensors");
    fs::write(&short_prefix, [1, 2, 3, 4, 5, 6, 7]).unwrap();
    assert_eq!(
        SafetensorsCatalog::open(short_prefix).unwrap_err().code(),
        ImportErrorCode::HeaderPrefixTruncated
    );

    let huge = dir.path().join("huge.safetensors");
    fs::write(&huge, (EXPECTED_MAX_HEADER_BYTES + 8).to_le_bytes()).unwrap();
    assert_eq!(
        SafetensorsCatalog::open(huge).unwrap_err().code(),
        ImportErrorCode::HeaderTooLarge
    );

    let impossible = dir.path().join("u64-max.safetensors");
    fs::write(&impossible, u64::MAX.to_le_bytes()).unwrap();
    assert_eq!(
        SafetensorsCatalog::open(impossible).unwrap_err().code(),
        ImportErrorCode::HeaderTooLarge
    );

    let unaligned = dir.path().join("unaligned.safetensors");
    let mut bytes = 7_u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(b"{}     ");
    fs::write(&unaligned, bytes).unwrap();
    assert_eq!(
        SafetensorsCatalog::open(unaligned).unwrap_err().code(),
        ImportErrorCode::HeaderLengthNotAligned
    );

    let truncated = dir.path().join("truncated.safetensors");
    let mut bytes = 16_u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(b"{}      ");
    fs::write(&truncated, bytes).unwrap();
    assert_eq!(
        SafetensorsCatalog::open(truncated).unwrap_err().code(),
        ImportErrorCode::HeaderTruncated
    );
}

#[test]
fn rejects_invalid_utf8_json_and_duplicate_object_keys() {
    let (_dir, path) = fixture(&[b'{', 0xff, b'}'], &[]);
    assert_eq!(
        SafetensorsCatalog::open(path).unwrap_err().code(),
        ImportErrorCode::InvalidHeaderUtf8
    );

    let (_dir, path) = fixture(br#"{"a":}"#, &[]);
    assert_eq!(
        SafetensorsCatalog::open(path).unwrap_err().code(),
        ImportErrorCode::InvalidHeaderJson
    );

    let (_dir, path) = fixture(
        br#"{"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#,
        &[0, 0],
    );
    assert_eq!(
        SafetensorsCatalog::open(path).unwrap_err().code(),
        ImportErrorCode::DuplicateHeaderKey
    );

    let (_dir, path) = fixture(
        br#"{"a":{"dtype":"BF16","dtype":"F16","shape":[1],"data_offsets":[0,2]}}"#,
        &[0, 0],
    );
    assert_eq!(
        SafetensorsCatalog::open(path).unwrap_err().code(),
        ImportErrorCode::DuplicateHeaderKey
    );
}

#[test]
fn rejects_unknown_fields_non_string_metadata_and_unsupported_dtypes() {
    let (_dir, path) = fixture(
        br#"{"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2],"surprise":true}}"#,
        &[0, 0],
    );
    assert_eq!(
        SafetensorsCatalog::open(path).unwrap_err().code(),
        ImportErrorCode::InvalidTensorMetadata
    );

    let (_dir, path) = fixture(br#"{"__metadata__":{"format":7}}"#, &[]);
    assert_eq!(
        SafetensorsCatalog::open(path).unwrap_err().code(),
        ImportErrorCode::InvalidMetadata
    );

    let (_dir, path) = fixture(
        br#"{"a":{"dtype":"F128","shape":[1],"data_offsets":[0,16]}}"#,
        &[0; 16],
    );
    assert_eq!(
        SafetensorsCatalog::open(path).unwrap_err().code(),
        ImportErrorCode::UnsupportedDtype
    );
}

#[test]
fn rejects_overflow_wrong_byte_counts_gaps_overlaps_and_trailing_data() {
    let (_dir, path) = fixture(
        br#"{"a":{"dtype":"BF16","shape":[18446744073709551615,2],"data_offsets":[0,0]}}"#,
        &[],
    );
    assert_eq!(
        SafetensorsCatalog::open(path).unwrap_err().code(),
        ImportErrorCode::ShapeElementCountOverflow
    );

    let (_dir, path) = fixture(
        br#"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,2]}}"#,
        &[0, 0],
    );
    assert_eq!(
        SafetensorsCatalog::open(path).unwrap_err().code(),
        ImportErrorCode::TensorByteLengthMismatch
    );

    let (_dir, path) = fixture(
        br#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}}"#,
        &[0, 0, 0],
    );
    assert_eq!(
        SafetensorsCatalog::open(path).unwrap_err().code(),
        ImportErrorCode::OverlappingData
    );

    let (_dir, path) = fixture(
        br#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[2,3]}}"#,
        &[0, 0, 0],
    );
    assert_eq!(
        SafetensorsCatalog::open(path).unwrap_err().code(),
        ImportErrorCode::NonContiguousData
    );

    let (_dir, path) = fixture(
        br#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
        &[0, 1],
    );
    assert_eq!(
        SafetensorsCatalog::open(path).unwrap_err().code(),
        ImportErrorCode::DataLengthMismatch
    );
}

#[test]
fn opening_catalog_does_not_materialize_a_large_sparse_payload() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sparse.safetensors");
    let header = padded_header(
        br#"{"large":{"dtype":"U8","shape":[67108864],"data_offsets":[0,67108864]}}"#,
    );
    let mut file = File::create(&path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.set_len(8 + header.len() as u64 + 67_108_864).unwrap();
    drop(file);

    let catalog = SafetensorsCatalog::open(path).expect("header-only import must succeed");
    assert_eq!(catalog.tensor("large").unwrap().byte_len(), 67_108_864);
}

#[test]
fn reader_api_proves_open_is_header_only_and_copying_is_bounded_streaming() {
    assert_eq!(COPY_BUFFER_BYTES, EXPECTED_COPY_BUFFER_BYTES);
    let tensor_len = EXPECTED_COPY_BUFFER_BYTES * 2 + 17;
    let prefix: Vec<u8> = (0..257).map(|i| (i * 17 + 3) as u8).collect();
    let tensor: Vec<u8> = (0..tensor_len)
        .map(|i| ((i * 131 + i / 251 + 29) % 256) as u8)
        .collect();
    let suffix: Vec<u8> = (0..31).map(|i| (255 - i * 5) as u8).collect();
    let tensor_start = prefix.len();
    let tensor_end = tensor_start + tensor.len();
    let body: Vec<u8> = prefix
        .iter()
        .chain(&tensor)
        .chain(&suffix)
        .copied()
        .collect();
    let header_json = format!(
        concat!(
            "{{",
            "\"suffix\":{{\"dtype\":\"U8\",\"shape\":[{}],\"data_offsets\":[{},{}]}},",
            "\"large\":{{\"dtype\":\"U8\",\"shape\":[{}],\"data_offsets\":[{},{}]}},",
            "\"prefix\":{{\"dtype\":\"U8\",\"shape\":[{}],\"data_offsets\":[0,{}]}}",
            "}}"
        ),
        suffix.len(),
        tensor_end,
        body.len(),
        tensor_len,
        tensor_start,
        tensor_end,
        prefix.len(),
        tensor_start,
    );
    let bytes = container_bytes(header_json.as_bytes(), &body);
    let file_len = bytes.len() as u64;
    let header_end = 8 + u64::from_le_bytes(bytes[..8].try_into().unwrap());
    let mut reader = InstrumentedReader::new(bytes.clone(), Some(header_end));

    let catalog = SafetensorsCatalog::read_from(&mut reader, file_len)
        .expect("open must never request payload bytes");
    assert_eq!(reader.total_read, header_end);
    assert_eq!(
        catalog.tensor("large").unwrap().byte_len(),
        tensor_len as u64
    );

    reader.allow_all_reads_and_reset_metrics();
    let mut copied_bytes = Vec::new();
    let copied = catalog
        .copy_tensor_from("large", &mut reader, &mut copied_bytes)
        .expect("bounded streaming copy");
    assert_eq!(copied, tensor_len as u64);
    assert_eq!(reader.total_read, tensor_len as u64, "must not over-read");
    assert_eq!(copied_bytes, tensor, "must seek to the exact tensor offset");
    assert!(reader.max_requested_read > 0);
    assert!(
        reader.max_requested_read <= EXPECTED_COPY_BUFFER_BYTES,
        "copy requested {} bytes in one Read call",
        reader.max_requested_read
    );

    // A reader-size limit alone is insufficient: a broken implementation could
    // accumulate every 64 KiB chunk in a Vec and only then write. Backpressure
    // makes that observable. After the writer's first partial 97-byte write and
    // immediate failure, no more than the first input chunk may have been read.
    let mut backpressured_reader = InstrumentedReader::new(bytes, None);
    let mut failing_writer = PartialThenFailWriter::default();
    let error = catalog
        .copy_tensor_from("large", &mut backpressured_reader, &mut failing_writer)
        .unwrap_err();
    assert_eq!(error.code(), ImportErrorCode::Io);
    assert_eq!(failing_writer.calls, 2);
    assert_eq!(failing_writer.accepted, tensor[..97]);
    assert!(backpressured_reader.total_read > 0);
    assert!(
        backpressured_reader.total_read <= EXPECTED_COPY_BUFFER_BYTES as u64,
        "reader consumed {} bytes before observing writer failure",
        backpressured_reader.total_read
    );
    assert!(backpressured_reader.total_read < tensor_len as u64);
}

#[cfg(unix)]
#[test]
fn rejects_symlinks_and_non_regular_files_before_opening_them() {
    use std::os::unix::fs::symlink;
    use std::process::Command;

    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("target.safetensors");
    write_container(&target, br#"{}"#, &[]);
    let link = dir.path().join("link.safetensors");
    symlink(&target, &link).unwrap();
    assert_eq!(
        SafetensorsCatalog::open(link).unwrap_err().code(),
        ImportErrorCode::NotRegularFile
    );

    let fifo = dir.path().join("pipe.safetensors");
    let status = Command::new("mkfifo").arg(&fifo).status().unwrap();
    assert!(status.success());
    assert_eq!(
        SafetensorsCatalog::open(fifo).unwrap_err().code(),
        ImportErrorCode::NotRegularFile
    );
}
