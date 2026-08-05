use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use pvlc_model_schema::TensorDtype;
use pvlc_safetensors::{
    COPY_BUFFER_BYTES, FP16_CHECKPOINT_CONVERSION_ID, Fp16CheckpointErrorCode, SafetensorsCatalog,
    convert_bf16_checkpoint_to_f16, convert_bf16_checkpoint_to_f16_stream, finite_bf16_to_f16_bits,
    stream_bf16_payload_to_f16,
};
use tempfile::TempDir;

#[path = "support/stream_instrumentation.rs"]
mod stream_instrumentation;
use stream_instrumentation::{InstrumentedReader, PartialThenFailWriter};

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

fn bf16_body(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|bits| bits.to_le_bytes()).collect()
}

fn fixture() -> (TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.safetensors");
    let mut body = bf16_body(&[0x0000, 0x8000, 0x3f80, 0xbf80]);
    body.extend_from_slice(&bf16_body(&[0x3880, 0x3800, 0x3380, 0x33c0, 0x477f]));
    write_container(
        &source,
        br#"{"__metadata__":{"format":"pt","owner":"contract"},"z":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]},"a":{"dtype":"BF16","shape":[5],"data_offsets":[8,18]}}"#,
        &body,
    );
    (dir, source)
}

#[test]
fn finite_bf16_values_convert_to_ieee_f16_with_ties_to_even() {
    for (source, expected) in [
        (0x0000, 0x0000), // +0
        (0x8000, 0x8000), // -0
        (0x3f80, 0x3c00), // +1
        (0xbf80, 0xbc00), // -1
        (0x3880, 0x0400), // minimum normal f16
        (0x3800, 0x0200), // exact subnormal
        (0x3380, 0x0001), // minimum subnormal
        (0x3300, 0x0000), // half-minimum tie rounds to even zero
        (0x33c0, 0x0002), // 1.5 minimum-subnormal tie rounds to even two
        (0xb3c0, 0x8002), // symmetric negative tie
        (0x477f, 0x7bf8), // largest finite BF16 below f16 overflow
    ] {
        assert_eq!(
            finite_bf16_to_f16_bits(source),
            Some(expected),
            "BF16 {source:#06x}"
        );
    }

    for rejected in [0x4780, 0xc780, 0x7f80, 0xff80, 0x7fc1, 0xffc1] {
        assert_eq!(
            finite_bf16_to_f16_bits(rejected),
            None,
            "BF16 {rejected:#06x} must not become a nonfinite f16 weight"
        );
    }
}

#[test]
fn checkpoint_conversion_is_deterministic_self_describing_and_loadable() {
    let (dir, source) = fixture();
    let first = dir.path().join("first.safetensors");
    let second = dir.path().join("second.safetensors");
    let source_before = fs::read(&source).unwrap();

    let first_report = convert_bf16_checkpoint_to_f16(&source, &first).unwrap();
    let second_report = convert_bf16_checkpoint_to_f16(&source, &second).unwrap();

    assert_eq!(fs::read(&source).unwrap(), source_before);
    assert_eq!(fs::read(&first).unwrap(), fs::read(&second).unwrap());
    assert_eq!(first_report, second_report);
    assert_eq!(first_report.conversion, FP16_CHECKPOINT_CONVERSION_ID);
    assert_eq!(first_report.tensor_count, 2);
    assert_eq!(first_report.element_count, 9);
    assert_eq!(first_report.source_bytes, source_before.len() as u64);
    assert_eq!(
        first_report.output_bytes,
        fs::metadata(&first).unwrap().len()
    );
    assert_eq!(
        first_report.source_blake3,
        blake3::hash(&source_before).to_hex().to_string(),
        "source identity must be independently reproducible from the whole file"
    );
    assert_eq!(
        first_report.output_blake3,
        blake3::hash(&fs::read(&first).unwrap())
            .to_hex()
            .to_string(),
        "output identity must be independently reproducible from the whole file"
    );
    assert_ne!(first_report.source_blake3, first_report.output_blake3);
    assert!(first_report.max_payload_buffer_bytes <= COPY_BUFFER_BYTES);

    let output = SafetensorsCatalog::open(&first).unwrap();
    assert_eq!(output.tensors().len(), 2);
    assert!(
        output
            .tensors()
            .iter()
            .all(|tensor| tensor.dtype == TensorDtype::Float16)
    );
    assert_eq!(output.tensor("a").unwrap().shape, [5]);
    assert_eq!(output.tensor("z").unwrap().shape, [4]);
    assert_eq!(output.tensor("z").unwrap().data_offsets, [0, 8]);
    assert_eq!(output.tensor("a").unwrap().data_offsets, [8, 18]);
    assert_eq!(
        output.metadata().get("format").map(String::as_str),
        Some("pt")
    );
    assert_eq!(
        output.metadata().get("owner").map(String::as_str),
        Some("contract")
    );
    assert_eq!(
        output.metadata().get("pvlc.conversion").map(String::as_str),
        Some(FP16_CHECKPOINT_CONVERSION_ID)
    );
    assert_eq!(
        output
            .metadata()
            .get("pvlc.source_blake3")
            .map(String::as_str),
        Some(first_report.source_blake3.as_str())
    );

    let mut raw_a = Vec::new();
    output.copy_tensor_to("a", &mut raw_a).unwrap();
    assert_eq!(
        raw_a,
        [0x0400_u16, 0x0200, 0x0001, 0x0002, 0x7bf8]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        output
            .load_tensor_f32("z")
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [
            0.0_f32.to_bits(),
            (-0.0_f32).to_bits(),
            1.0_f32.to_bits(),
            (-1.0_f32).to_bits(),
        ]
    );
}

#[test]
fn conversion_streams_with_bounded_reads_and_observes_writer_backpressure() {
    let element_count = COPY_BUFFER_BYTES + 1;
    let source_bits = [0x0000_u16, 0x3f80, 0xbf80, 0x3880, 0x3380];
    let payload = (0..element_count)
        .flat_map(|index| source_bits[index % source_bits.len()].to_le_bytes())
        .collect::<Vec<_>>();

    let mut reader = InstrumentedReader::new(payload.clone(), None);
    let mut output = Vec::new();
    let report =
        stream_bf16_payload_to_f16(&mut reader, &mut output, element_count as u64, "weight")
            .unwrap();
    assert_eq!(report.element_count, element_count as u64);
    assert_eq!(output.len(), payload.len());
    assert_eq!(
        reader.total_read,
        payload.len() as u64,
        "must not over-read"
    );
    assert!(reader.max_requested_read > 0);
    assert!(
        reader.max_requested_read <= COPY_BUFFER_BYTES,
        "converter requested {} input bytes in one read",
        reader.max_requested_read
    );
    assert_eq!(report.max_buffer_bytes, reader.max_requested_read);

    // A bounded Read size is not enough: an implementation could accumulate
    // every chunk before writing. The second write fails after a partial
    // 97-byte acceptance, so the converter must stop before consuming the
    // second input chunk.
    let mut backpressured_reader = InstrumentedReader::new(payload.clone(), None);
    let mut failing_writer = PartialThenFailWriter::default();
    let error = stream_bf16_payload_to_f16(
        &mut backpressured_reader,
        &mut failing_writer,
        element_count as u64,
        "weight",
    )
    .unwrap_err();
    assert_eq!(error.code(), Fp16CheckpointErrorCode::Io);
    assert_eq!(error.tensor_name(), Some("weight"));
    assert_eq!(failing_writer.calls, 2);
    assert_eq!(failing_writer.accepted.len(), 97);
    assert!(backpressured_reader.total_read > 0);
    assert!(
        backpressured_reader.total_read <= COPY_BUFFER_BYTES as u64,
        "converter read {} bytes before observing writer failure",
        backpressured_reader.total_read
    );
    assert!(backpressured_reader.total_read < payload.len() as u64);
}

#[test]
fn whole_checkpoint_stream_engine_uses_the_same_bounded_payload_authority() {
    let element_count = COPY_BUFFER_BYTES + 1;
    let byte_count = element_count * 2;
    let header = format!(
        r#"{{"weight":{{"dtype":"BF16","shape":[{element_count}],"data_offsets":[0,{byte_count}]}}}}"#
    );
    let payload = (0..element_count)
        .flat_map(|index| [0x0000_u16, 0x3f80, 0xbf80][index % 3].to_le_bytes())
        .collect::<Vec<_>>();
    let source = container_bytes(header.as_bytes(), &payload);
    let mut reader = InstrumentedReader::new(source.clone(), None);
    let mut output = Vec::new();

    let report =
        convert_bf16_checkpoint_to_f16_stream(&mut reader, source.len() as u64, &mut output)
            .unwrap();
    assert_eq!(report.element_count, element_count as u64);
    assert_eq!(
        report.source_blake3,
        blake3::hash(&source).to_hex().to_string()
    );
    assert_eq!(
        report.output_blake3,
        blake3::hash(&output).to_hex().to_string()
    );
    assert_eq!(report.max_payload_buffer_bytes, COPY_BUFFER_BYTES);
    assert!(
        reader.max_requested_read <= COPY_BUFFER_BYTES,
        "whole-checkpoint engine requested {} bytes in one read",
        reader.max_requested_read
    );

    const SOURCE: &str = include_str!("../src/lib.rs");
    let wrapper = SOURCE
        .split("pub fn convert_bf16_checkpoint_to_f16(")
        .nth(1)
        .and_then(|tail| tail.split("\n}\n\n").next())
        .expect("path conversion wrapper body");
    assert!(
        wrapper.contains("convert_bf16_checkpoint_to_f16_stream("),
        "path wrapper bypasses the bounded whole-checkpoint stream authority"
    );
    for forbidden in [
        "fs::read(",
        "std::fs::read(",
        ".read_to_end(",
        ".read_to_string(",
    ] {
        assert!(
            !wrapper.contains(forbidden),
            "path wrapper can materialize the whole checkpoint via {forbidden}"
        );
    }
}

#[test]
fn conversion_rejects_dtype_and_value_drift_without_leaving_partial_output() {
    let dir = tempfile::tempdir().unwrap();
    let cases: [(&str, &str, u16, Fp16CheckpointErrorCode); 5] = [
        (
            "wrong-dtype",
            "F16",
            0x3c00_u16,
            Fp16CheckpointErrorCode::UnsupportedSourceDtype,
        ),
        (
            "positive-infinity",
            "BF16",
            0x7f80_u16,
            Fp16CheckpointErrorCode::NonFiniteSource,
        ),
        (
            "negative-infinity",
            "BF16",
            0xff80_u16,
            Fp16CheckpointErrorCode::NonFiniteSource,
        ),
        (
            "nan",
            "BF16",
            0x7fc1_u16,
            Fp16CheckpointErrorCode::NonFiniteSource,
        ),
        (
            "overflow",
            "BF16",
            0x4780_u16,
            Fp16CheckpointErrorCode::OutOfRangeSource,
        ),
    ];
    for (name, dtype, bits, code) in cases {
        let source = dir.path().join(format!("{name}.safetensors"));
        let output = dir.path().join(format!("{name}.fp16.safetensors"));
        write_container(
            &source,
            format!(r#"{{"weight":{{"dtype":"{dtype}","shape":[1],"data_offsets":[0,2]}}}}"#)
                .as_bytes(),
            &bits.to_le_bytes(),
        );
        let error = convert_bf16_checkpoint_to_f16(&source, &output).unwrap_err();
        assert_eq!(error.code(), code);
        assert_eq!(error.tensor_name(), Some("weight"));
        if code != Fp16CheckpointErrorCode::UnsupportedSourceDtype {
            assert_eq!(error.element_index(), Some(0));
        }
        assert!(!output.exists(), "failed conversion left {output:?}");
    }
}

#[test]
fn late_semantic_failure_removes_the_staged_file_after_streaming_a_full_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("late-failure.safetensors");
    let output = dir.path().join("late-failure.fp16.safetensors");
    let element_count = COPY_BUFFER_BYTES / 2 + 1;
    let byte_count = element_count * 2;
    let header = format!(
        r#"{{"weight":{{"dtype":"BF16","shape":[{element_count}],"data_offsets":[0,{byte_count}]}}}}"#
    );
    let mut body = vec![0_u8; byte_count];
    body[byte_count - 2..].copy_from_slice(&0x7f80_u16.to_le_bytes());
    write_container(&source, header.as_bytes(), &body);

    let error = convert_bf16_checkpoint_to_f16(&source, &output).unwrap_err();
    assert_eq!(error.code(), Fp16CheckpointErrorCode::NonFiniteSource);
    assert_eq!(error.tensor_name(), Some("weight"));
    assert_eq!(error.element_index(), Some(element_count as u64 - 1));
    assert!(!output.exists());

    let mut remaining = fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect::<Vec<_>>();
    remaining.sort();
    assert_eq!(
        remaining,
        [source.file_name().unwrap().to_owned()],
        "atomic conversion leaked a staged output after a late failure"
    );
}

#[test]
fn conversion_refuses_aliasing_and_existing_destinations_without_overwrite() {
    let (dir, source) = fixture();
    let alias_error = convert_bf16_checkpoint_to_f16(&source, &source).unwrap_err();
    assert_eq!(
        alias_error.code(),
        Fp16CheckpointErrorCode::SourceDestinationAlias
    );

    let output = dir.path().join("existing.safetensors");
    fs::write(&output, b"sentinel").unwrap();
    let error = convert_bf16_checkpoint_to_f16(&source, &output).unwrap_err();
    assert_eq!(error.code(), Fp16CheckpointErrorCode::OutputAlreadyExists);
    assert_eq!(fs::read(output).unwrap(), b"sentinel");
}
