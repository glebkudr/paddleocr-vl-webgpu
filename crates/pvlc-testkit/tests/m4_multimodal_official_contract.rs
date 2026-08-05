use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use pvlc_cpu_ref::{
    assemble_multimodal_embeddings_f32, decode_mrope_position_ids, image_placeholder_count,
    mrope_position_ids,
};
use pvlc_safetensors::SafetensorsCatalog;

const MODEL_REVISION: &str = "66317acc4c9fc17bd154591ce650735cd2855f3e";
const GOLDEN_LOCK: &str = include_str!("../../../goldens/golden.lock");
const GOLDEN_LOCK_BLAKE3: &str = "40947f87eec2ac0f75ce671ca9226bb335adbe6254e5c1858f5c2ae6310450c9";
const IMAGE_TOKEN_ID: u32 = 100_295;
const VISION_START_TOKEN_ID: u32 = 101_305;
const HIDDEN_SIZE: usize = 1_024;
const VOCAB_SIZE: usize = 103_424;
const SPATIAL_MERGE_SIZE: usize = 2;

// Pinned revision PaddleOCRVLForConditionalGeneration.get_rope_index emits [3, 1, seq].
// These anchors hash that tensor axis-major as raw little-endian i64, with no JSON framing.
#[derive(Clone, Copy)]
struct OfficialCase {
    directory: &'static str,
    artifact_path: &'static str,
    case_id: &'static str,
    trace_level: &'static str,
    bundle_digest: &'static str,
    semantic_fingerprint: &'static str,
    grid: [usize; 3],
    sequence_length: usize,
    placeholder_count: usize,
    mrope_blake3: &'static str,
    rope_delta: i64,
}

const CASES: [OfficialCase; 2] = [
    OfficialCase {
        directory: "ocr.clean_latin.0001-l3",
        artifact_path: "artifacts/goldens/ocr.clean_latin.0001-l3",
        case_id: "ocr.clean_latin.0001",
        trace_level: "L3",
        bundle_digest: "blake3:35572ac07da5fddee97becb46fb866f906200f433f4c5677ad20fdf36440acf9",
        semantic_fingerprint: "blake3:632cbe2de6f47f8f84c764e3a551bf99f352d7459c98d939eb89725357f952b4",
        grid: [1, 22, 58],
        sequence_length: 332,
        placeholder_count: 319,
        mrope_blake3: "b51cce87812eafc3316d2606374eb1b9690db1286c9f418d7ca488b75d4c843b",
        rope_delta: -290,
    },
    OfficialCase {
        directory: "table.simple.0001-l2",
        artifact_path: "artifacts/goldens/table.simple.0001-l2",
        case_id: "table.simple.0001",
        trace_level: "L2",
        bundle_digest: "blake3:4cecc8b2abc37030920acf8860bb317084125dbbbcae5af394fccd0c8044c842",
        semantic_fingerprint: "blake3:91f72ffdf81070c2c9ca31628605064bdc02bf19417482cefce41d6a27aa5404",
        grid: [1, 30, 58],
        sequence_length: 448,
        placeholder_count: 435,
        mrope_blake3: "fb80a718233a1a038a3f09e519cb1b000293394ea8a75abe541f86fbbd0740d8",
        rope_delta: -406,
    },
];

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn require_model_checkpoint() -> Option<PathBuf> {
    let path = repository()
        .join("models/snapshots")
        .join(MODEL_REVISION)
        .join("model.safetensors");
    if path.is_file() {
        return Some(path);
    }
    assert_ne!(
        std::env::var("PVLC_REQUIRE_MODEL").as_deref(),
        Ok("1"),
        "PVLC_REQUIRE_MODEL=1 but pinned checkpoint is absent at {}",
        path.display()
    );
    eprintln!(
        "skipping local multimodal assembly oracle: {}",
        path.display()
    );
    None
}

fn assert_exact_golden_identity(case: OfficialCase) {
    assert_eq!(
        blake3::hash(GOLDEN_LOCK.as_bytes()).to_hex().as_str(),
        GOLDEN_LOCK_BLAKE3
    );
    let lock = toml::from_str::<toml::Table>(GOLDEN_LOCK).unwrap();
    assert_eq!(lock["format_version"].as_integer(), Some(1));
    assert_eq!(lock["model_revision"].as_str(), Some(MODEL_REVISION));
    assert_eq!(lock["trace_schema_version"].as_integer(), Some(1));
    let matches = lock["bundles"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|bundle| bundle["case_id"].as_str() == Some(case.case_id))
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1, "case_id={}", case.case_id);
    let bundle = matches[0];
    assert_eq!(bundle["trace_level"].as_str(), Some(case.trace_level));
    assert_eq!(bundle["artifact_path"].as_str(), Some(case.artifact_path));
    assert_eq!(bundle["bundle_digest"].as_str(), Some(case.bundle_digest));
    assert_eq!(
        bundle["semantic_fingerprint"].as_str(),
        Some(case.semantic_fingerprint)
    );
}

fn read_tensor_bytes(
    catalog: &SafetensorsCatalog,
    path: &Path,
    name: &str,
    expected_shape: &[u64],
    expected_dtype: &str,
) -> Vec<u8> {
    let tensor = catalog
        .tensor(name)
        .unwrap_or_else(|| panic!("missing tensor {name}"));
    assert_eq!(tensor.shape, expected_shape, "tensor={name}");
    assert_eq!(
        tensor.dtype.safetensors_name(),
        expected_dtype,
        "tensor={name}"
    );
    let absolute_start = 8_u64
        .checked_add(catalog.header_len())
        .and_then(|start| start.checked_add(tensor.data_offsets[0]))
        .unwrap();
    let byte_len = usize::try_from(tensor.byte_len()).unwrap();
    let mut bytes = vec![0_u8; byte_len];
    let mut file = File::open(path).unwrap();
    file.seek(SeekFrom::Start(absolute_start)).unwrap();
    file.read_exact(&mut bytes).unwrap();
    bytes
}

fn load_i64_tensor(
    catalog: &SafetensorsCatalog,
    path: &Path,
    name: &str,
    expected_shape: &[u64],
) -> Vec<i64> {
    read_tensor_bytes(catalog, path, name, expected_shape, "I64")
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn load_processor(case: OfficialCase) -> (Vec<u32>, Vec<u8>, [usize; 3]) {
    let path = repository()
        .join("artifacts/goldens")
        .join(case.directory)
        .join("processor.safetensors");
    let catalog = SafetensorsCatalog::open(&path).unwrap();
    let input_ids = load_i64_tensor(
        &catalog,
        &path,
        "processor.input_ids",
        &[1, case.sequence_length as u64],
    )
    .into_iter()
    .map(|value| u32::try_from(value).unwrap())
    .collect::<Vec<_>>();
    let attention_mask = load_i64_tensor(
        &catalog,
        &path,
        "processor.attention_mask",
        &[1, case.sequence_length as u64],
    )
    .into_iter()
    .map(|value| u8::try_from(value).unwrap())
    .collect::<Vec<_>>();
    let grid = load_i64_tensor(&catalog, &path, "processor.image_grid_thw", &[1, 3])
        .into_iter()
        .map(|value| usize::try_from(value).unwrap())
        .collect::<Vec<_>>();
    (input_ids, attention_mask, grid.try_into().unwrap())
}

fn gather_bf16_embedding_rows(
    catalog: &SafetensorsCatalog,
    model_path: &Path,
    input_ids: &[u32],
) -> Vec<f32> {
    let tensor = catalog
        .tensor("model.embed_tokens.weight")
        .expect("pinned token embedding tensor");
    assert_eq!(tensor.dtype.safetensors_name(), "BF16");
    assert_eq!(tensor.shape, [VOCAB_SIZE as u64, HIDDEN_SIZE as u64]);
    let row_bytes = HIDDEN_SIZE.checked_mul(2).unwrap();
    assert_eq!(tensor.byte_len(), (VOCAB_SIZE * row_bytes) as u64);
    let tensor_start = 8_u64
        .checked_add(catalog.header_len())
        .and_then(|start| start.checked_add(tensor.data_offsets[0]))
        .unwrap();
    let mut file = File::open(model_path).unwrap();
    let mut row = vec![0_u8; row_bytes];
    let mut output = Vec::with_capacity(input_ids.len() * HIDDEN_SIZE);
    for &token_id in input_ids {
        let token = usize::try_from(token_id).unwrap();
        assert!(token < VOCAB_SIZE);
        let row_offset = token.checked_mul(row_bytes).unwrap() as u64;
        file.seek(SeekFrom::Start(tensor_start + row_offset))
            .unwrap();
        file.read_exact(&mut row).unwrap();
        output.extend(row.chunks_exact(2).map(|chunk| {
            let bits = u16::from_le_bytes(chunk.try_into().unwrap());
            f32::from_bits(u32::from(bits) << 16)
        }));
    }
    assert_eq!(output.len(), input_ids.len() * HIDDEN_SIZE);
    assert!((input_ids.len() * row_bytes) as u64 <= tensor.byte_len());
    output
}

fn hash_i64_le(positions: &[Vec<i64>; 3]) -> String {
    let mut hasher = blake3::Hasher::new();
    for axis in positions {
        for value in axis {
            hasher.update(&value.to_le_bytes());
        }
    }
    hasher.finalize().to_hex().to_string()
}

fn rows_differ(left: &[f32], right: &[f32], width: usize) -> Vec<usize> {
    assert_eq!(left.len(), right.len());
    assert!(width > 0 && left.len().is_multiple_of(width));
    left.chunks_exact(width)
        .zip(right.chunks_exact(width))
        .enumerate()
        .filter_map(|(row, (left, right))| {
            left.iter()
                .zip(right)
                .any(|(left, right)| left.to_bits() != right.to_bits())
                .then_some(row)
        })
        .collect()
}

fn assert_bits_equal(left: &[f32], right: &[f32], label: &str) {
    assert_eq!(left.len(), right.len(), "{label}");
    assert!(
        left.iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits()),
        "{label} differs"
    );
}

#[test]
fn pinned_profiles_have_exact_placeholder_mrope_and_first_decode_anchors() {
    if CASES.iter().any(|case| {
        !repository()
            .join("artifacts/goldens")
            .join(case.directory)
            .join("processor.safetensors")
            .is_file()
    }) {
        eprintln!("skipping official multimodal anchors: processor fixtures are not distributed");
        return;
    }
    for case in CASES {
        assert_exact_golden_identity(case);
        let (input_ids, attention_mask, grid) = load_processor(case);
        assert_eq!(grid, case.grid);
        assert!(attention_mask.iter().all(|value| *value == 1));
        assert_eq!(
            input_ids
                .iter()
                .filter(|token| **token == IMAGE_TOKEN_ID)
                .count(),
            case.placeholder_count
        );
        assert_eq!(
            input_ids
                .iter()
                .filter(|token| **token == VISION_START_TOKEN_ID)
                .count(),
            1
        );
        assert_eq!(
            image_placeholder_count(&[grid], SPATIAL_MERGE_SIZE).unwrap(),
            case.placeholder_count
        );

        let positions = mrope_position_ids(
            &input_ids,
            Some(&attention_mask),
            &[grid],
            IMAGE_TOKEN_ID,
            VISION_START_TOKEN_ID,
            SPATIAL_MERGE_SIZE,
        )
        .unwrap();
        assert_eq!(positions.rope_delta, case.rope_delta);
        assert!(
            positions
                .position_ids
                .iter()
                .all(|axis: &Vec<i64>| axis.len() == case.sequence_length)
        );
        assert_eq!(hash_i64_le(&positions.position_ids), case.mrope_blake3);

        let first_decode =
            decode_mrope_position_ids(case.sequence_length, 1, positions.rope_delta).unwrap();
        assert_eq!(first_decode, [vec![42], vec![42], vec![42]]);
    }
}

#[test]
fn direct_official_assembly_is_bit_exact_and_projected_row_mutations_are_isolated() {
    let Some(model_path) = require_model_checkpoint() else {
        return;
    };
    let model = SafetensorsCatalog::open(&model_path).unwrap();

    for case in CASES {
        assert_exact_golden_identity(case);
        let (input_ids, _attention_mask, grid) = load_processor(case);
        assert_eq!(grid, case.grid);
        let token_embeddings = gather_bf16_embedding_rows(&model, &model_path, &input_ids);

        let stage_path = repository()
            .join("artifacts/goldens")
            .join(case.directory)
            .join("stage-checkpoints.safetensors");
        let stage = SafetensorsCatalog::open(&stage_path).unwrap();
        assert_eq!(
            stage.tensor("projector.final").unwrap().shape,
            [case.placeholder_count as u64, HIDDEN_SIZE as u64]
        );
        assert_eq!(
            stage.tensor("multimodal.inputs_embeds").unwrap().shape,
            [1, case.sequence_length as u64, HIDDEN_SIZE as u64]
        );
        let projected = stage.load_tensor_f32("projector.final").unwrap();
        let expected = stage.load_tensor_f32("multimodal.inputs_embeds").unwrap();
        let assembled = assemble_multimodal_embeddings_f32(
            &token_embeddings,
            &projected,
            &input_ids,
            HIDDEN_SIZE,
            IMAGE_TOKEN_ID,
        )
        .unwrap();
        assert_bits_equal(&assembled, &expected, case.case_id);

        let image_rows = input_ids
            .iter()
            .enumerate()
            .filter_map(|(row, token)| (*token == IMAGE_TOKEN_ID).then_some(row))
            .collect::<Vec<_>>();
        assert_eq!(image_rows.len(), case.placeholder_count);
        for (row, token) in input_ids.iter().copied().enumerate() {
            let range = row * HIDDEN_SIZE..(row + 1) * HIDDEN_SIZE;
            if token == IMAGE_TOKEN_ID {
                let projected_row = image_rows.iter().position(|value| *value == row).unwrap();
                let projected_range =
                    projected_row * HIDDEN_SIZE..(projected_row + 1) * HIDDEN_SIZE;
                assert_bits_equal(
                    &assembled[range],
                    &projected[projected_range],
                    "projected row order",
                );
            } else {
                assert_bits_equal(
                    &assembled[range.clone()],
                    &token_embeddings[range],
                    "text row changed",
                );
            }
        }

        let poison_index = case.placeholder_count / 2;
        let mut poisoned = projected.clone();
        let poison_offset = poison_index * HIDDEN_SIZE;
        poisoned[poison_offset] = f32::from_bits(poisoned[poison_offset].to_bits() ^ 1);
        let poisoned_output = assemble_multimodal_embeddings_f32(
            &token_embeddings,
            &poisoned,
            &input_ids,
            HIDDEN_SIZE,
            IMAGE_TOKEN_ID,
        )
        .unwrap();
        assert_eq!(
            rows_differ(&assembled, &poisoned_output, HIDDEN_SIZE),
            [image_rows[poison_index]]
        );
        assert!(rows_differ(&expected, &poisoned_output, HIDDEN_SIZE).len() == 1);

        let last = case.placeholder_count - 1;
        let mut swapped = projected;
        for channel in 0..HIDDEN_SIZE {
            swapped.swap(channel, last * HIDDEN_SIZE + channel);
        }
        let swapped_output = assemble_multimodal_embeddings_f32(
            &token_embeddings,
            &swapped,
            &input_ids,
            HIDDEN_SIZE,
            IMAGE_TOKEN_ID,
        )
        .unwrap();
        assert_eq!(
            rows_differ(&assembled, &swapped_output, HIDDEN_SIZE),
            [image_rows[0], image_rows[last]]
        );
        assert!(rows_differ(&expected, &swapped_output, HIDDEN_SIZE).len() == 2);
    }
}
