use std::path::Path;

use pvlc_cpu_ref::{add_interpolated_position_embedding_f32, patch_projection_f32};
use pvlc_safetensors::SafetensorsCatalog;

const MODEL_REVISION: &str = "66317acc4c9fc17bd154591ce650735cd2855f3e";
const TOKENS: usize = 1_740;
const HIDDEN: usize = 1_152;
const CHANNELS: usize = 3;
const PATCH_SIZE: usize = 14;

fn require_model(path: &Path) {
    if !path.is_file() {
        if std::env::var("PVLC_REQUIRE_MODEL").as_deref() == Ok("1") {
            panic!("pinned model is required at {}", path.display());
        }
        eprintln!(
            "skipping L2 input oracle because {} is absent",
            path.display()
        );
    }
}

#[test]
#[ignore = "materializes the full 1740 x 1152 L2 patch-plus-position input oracle"]
fn pinned_table_l2_patch_plus_position_input_has_reviewed_anchor() {
    assert_eq!(std::env::var("PVLC_REQUIRE_MODEL").as_deref(), Ok("1"));
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model_path = root
        .join("models/snapshots")
        .join(MODEL_REVISION)
        .join("model.safetensors");
    require_model(&model_path);
    let model = SafetensorsCatalog::open(model_path).unwrap();
    let processor = SafetensorsCatalog::open(
        root.join("artifacts/goldens/table.simple.0001-l2/processor.safetensors"),
    )
    .unwrap();

    let pixels = processor.load_tensor_f32("processor.pixel_values").unwrap();
    let weights = model
        .load_tensor_f32("visual.vision_model.embeddings.patch_embedding.weight")
        .unwrap();
    let bias = model
        .load_tensor_f32("visual.vision_model.embeddings.patch_embedding.bias")
        .unwrap();
    let positions = model
        .load_tensor_f32("visual.vision_model.embeddings.position_embedding.weight")
        .unwrap();
    let patches = patch_projection_f32(
        &pixels, TOKENS, CHANNELS, PATCH_SIZE, &weights, &bias, HIDDEN,
    )
    .unwrap();
    let embedding = add_interpolated_position_embedding_f32(
        &patches,
        HIDDEN,
        &positions,
        27,
        27,
        &[[1, 30, 58]],
    )
    .unwrap();
    let bytes = embedding
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    assert_eq!(bytes.len(), 8_017_920);
    assert_eq!(
        blake3::hash(&bytes).to_hex().as_str(),
        "645e12596caffcd4b394202a1b790acbb51d242cf6f616c0bade5d012eece742"
    );
}
