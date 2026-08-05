use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::Read,
    path::Path,
};

use pvlc_cpu_ref::{add_interpolated_position_embedding_f32, patch_projection_f32};
use pvlc_model_schema::{COMPILER_MODEL_ABI, MODEL_ID, MODEL_REVISION, TensorDtype};
use pvlc_pack::{
    DecoderWeightStorage, VISION_STACK_SHARD_MANIFEST_SCHEMA_VERSION, VisionStackShardDescriptor,
    VisionStackShardKind, VisionStackShardManifest, VisionStackShardOracle,
    canonical_vision_stack_shard_manifest_bytes,
};
use pvlc_safetensors::SafetensorsCatalog;

pub use pvlc_pack::OfficialVisionStackProfile;

use super::{
    ModelLock, SourceError, SourceErrorCode, is_lower_hex_64, same_snapshot, verify_model_source,
    write_atomic,
};

const HIDDEN: u32 = 1_152;
const HEADS: u32 = 16;
const HEAD_DIM: u32 = 72;
const INTERMEDIATE: u32 = 4_304;
const LAYERS: u32 = 27;
const LAYER_NORM_EPSILON: f32 = 1.0e-6;
const EXPECTED_CHECKPOINT_BYTES: u64 = 29_399_040;
const EXPECTED_CHECKPOINT_BLAKE3: &str =
    "6949b4d783f2a65f653e52f9f5dc29380834bb2c5eee5a7d646b07abd70c3f4a";
const TABLE_L2_INPUT_BYTES: u64 = 8_017_920;
const TABLE_L2_INPUT_BLAKE3: &str =
    "645e12596caffcd4b394202a1b790acbb51d242cf6f616c0bade5d012eece742";
const TABLE_L2_EXPECTED_BYTES: u64 = 8_017_920;
const TABLE_L2_EXPECTED_BLAKE3: &str =
    "fcd101b25a04e1b4e0984e5d712094630f11c22d4fc57abdf743e9fd7a79aed9";
const HASH_BUFFER_BYTES: usize = 256 * 1_024;

const OFFICIAL_SHARD_ANCHORS: [(&str, u64, &str); 29] = [
    (
        "input.embeddings",
        5_879_808,
        "f09f99ef13ada84b0c21621bf18ac1f54ae89798cba6fa20753d34cbad13b050",
    ),
    (
        "weights.vision_layer.00",
        60_958_016,
        "9da9e50460ebccd33567f7e28ba56d718ce086ae310bcf3e26383a350402dbe4",
    ),
    (
        "weights.vision_layer.01",
        60_958_016,
        "698e347e6d4c15004746db8a351d89ce05901d730c3732fc86e050398a36f4d8",
    ),
    (
        "weights.vision_layer.02",
        60_958_016,
        "50a8ec3a99b5032bb1967576213fe73d220b854cf17cabd248a7f06f88b4ccbe",
    ),
    (
        "weights.vision_layer.03",
        60_958_016,
        "01055c2903d3cacb81cba1ca0c2412ffa1825139bbce3e8ccdeb0becb8f54aaa",
    ),
    (
        "weights.vision_layer.04",
        60_958_016,
        "ea0e8978a3e7ff0f05e3bd1e3ca1473de41711b5942ca64b026fe4f2648c3f9d",
    ),
    (
        "weights.vision_layer.05",
        60_958_016,
        "2a86698487e95545fb4617b9b42377ec25cb746139514f49ffb228d3d26accdd",
    ),
    (
        "weights.vision_layer.06",
        60_958_016,
        "5741e0b7f1fc4b6dd228a5d79071d0102f36f369a67d98ab3f58733d1058bbf6",
    ),
    (
        "weights.vision_layer.07",
        60_958_016,
        "bd1fa09708a4547336a161ae60d27559671a29f4c2764e04e18598e1f372538b",
    ),
    (
        "weights.vision_layer.08",
        60_958_016,
        "8e904ac7d70cfef17ff63ab30758fe4ab9f99353132b0ec0d50a48804311776b",
    ),
    (
        "weights.vision_layer.09",
        60_958_016,
        "c352d572d7ae71e0b74a2e49e6a5e2474baf8d8be0d896eced2e1bb990fa3edc",
    ),
    (
        "weights.vision_layer.10",
        60_958_016,
        "873a78a5da2bd97268d217817dcae4420231ec1958f2f6683172e92d00377744",
    ),
    (
        "weights.vision_layer.11",
        60_958_016,
        "2e5ef92dfa2e42f768701f1279f892fa0331ded1f7337565b4695d2bcd581ad7",
    ),
    (
        "weights.vision_layer.12",
        60_958_016,
        "5c8164610eb986da7bc9a74a31602fcc956806e006f13ec111d1f3049c9f7f53",
    ),
    (
        "weights.vision_layer.13",
        60_958_016,
        "8876293649a3914a69384b2d7fa50403556bae07f179c1e78af1d09e3589876a",
    ),
    (
        "weights.vision_layer.14",
        60_958_016,
        "f80fc4eca453414edd2a45b7f30796c34630c694664c70417620729371233a37",
    ),
    (
        "weights.vision_layer.15",
        60_958_016,
        "677e6d3a6bcaf47d90d43d03f629cbd9a327b2eb054956b253c612787f731382",
    ),
    (
        "weights.vision_layer.16",
        60_958_016,
        "af57dff67d882698f09221b557049871cb4647e837d1b6cd0b62abda5b42765b",
    ),
    (
        "weights.vision_layer.17",
        60_958_016,
        "439bfdd1ebf51ee8e6253231a1055eee575a638146a182d333f3664553a55afe",
    ),
    (
        "weights.vision_layer.18",
        60_958_016,
        "7ca4ebaa8c7ffc522e85c59f652073e8cb367b5663d43574c1080bdc663d91ff",
    ),
    (
        "weights.vision_layer.19",
        60_958_016,
        "fa9e1021246b1e4c40ec9c65774608665784d97b64504f862561b98e06827037",
    ),
    (
        "weights.vision_layer.20",
        60_958_016,
        "2053feaf72d28f3dafd78a10ad1dd31cfe876e482a4c9da5a93a7b08e0243627",
    ),
    (
        "weights.vision_layer.21",
        60_958_016,
        "1eb3e6ab370cbc8fb00192ca53331f883aaf863c444e911f9e1473d0019a936f",
    ),
    (
        "weights.vision_layer.22",
        60_958_016,
        "edf5f6eb5de0745be6a167a5d277980f1c8ca4ab977539c322eb8ec58cc093c2",
    ),
    (
        "weights.vision_layer.23",
        60_958_016,
        "dc715073e1f6b61fd7d200d8847316f5ee34fb29b760dc0473434614460846e9",
    ),
    (
        "weights.vision_layer.24",
        60_958_016,
        "8acf8fbf8fd5652c92bcaa015fb93738016bb8f3e9d8a7962bb424ecb187f4fd",
    ),
    (
        "weights.vision_layer.25",
        60_958_016,
        "ff35c13695bdde8cdf80aee7c5a08468c2498ec0148557ddf465197798dd4b6e",
    ),
    (
        "weights.vision_layer.26",
        60_958_016,
        "0cd33fe73cf585a200695ae82ae9ee2d1b6e0051c342a98d7dfbe9677e2580b8",
    ),
    (
        "weights.vision_post_norm",
        9_216,
        "ded3c979a5a529f0cc5538b175cca3216f9f5412caac18f2639546b654f97086",
    ),
];

const PINNED_GOLDEN_FILES: [(&str, u64, &str); 10] = [
    (
        "case.json",
        275,
        "5653008e85ffc95fed86cb9b97d105db753aa603d6a61488ebd568ae9c8463da",
    ),
    (
        "deep-checkpoints.safetensors",
        81_205_664,
        "d366c971675a1a0425aa43bee3af410d02337e15efccedbe13da73976b55b85d",
    ),
    (
        "hashes.json",
        1_065,
        "35572ac07da5fddee97becb46fb866f906200f433f4c5677ad20fdf36440acf9",
    ),
    (
        "manifest.json",
        756,
        "c5e2cfce1908c400dd9da0eeac68377f0b80963518e5cbec4ebc08d84d614269",
    ),
    (
        "probes.bin",
        36_685,
        "2f7ce7404474b3c8a60ed2fa87ce460f66c7affd53871edbc1d07cbcdb2fd7dd",
    ),
    (
        "processor.safetensors",
        3_006_840,
        "5dbc3e12099c59e851c5277b2c370f0ec8dd21dfbc8a40beafdca3ac625e3c21",
    ),
    (
        "source-image.bin",
        23_492,
        "b4aea513d42e04afe06f351bc322208172252c4e33971ce95ff3b0ebd0d7ea73",
    ),
    (
        "stage-checkpoints.safetensors",
        4_480_368,
        "74e3eb04ffe8f47f9f46ee241b1ca918af23f2e057301ea3c63f897765eaa2f6",
    ),
    (
        "tensor-stats.jsonl",
        13_789,
        "108c98ecc1fc420a6cdb95615357c6e1012a94907d5754c87f473a0a90464883",
    ),
    (
        "token-trace.jsonl",
        1_679,
        "9bd5bead639b4e698e973c25ac93892130e5797d81463eb828f434ed8d624f62",
    ),
];

const PINNED_TABLE_L2_GOLDEN_FILES: [(&str, u64, &str); 9] = [
    (
        "case.json",
        288,
        "3200f192effb57f5e74098e18340027be5596ded37cca74096711f8898f98b04",
    ),
    (
        "hashes.json",
        938,
        "4cecc8b2abc37030920acf8860bb317084125dbbbcae5af394fccd0c8044c842",
    ),
    (
        "manifest.json",
        722,
        "26f2c04012ea7fb8d843fab69dd81b6fb149e4038c052f811c683437faa673e9",
    ),
    (
        "probes.bin",
        7_619,
        "02839886897543b8f005c311b17582c4beb7be36fb49990ae4bfa4b8ce3ae405",
    ),
    (
        "processor.safetensors",
        4_100_024,
        "c58bae89e79ca211159bd72d3dfc7405458725389e84c74cfcb14cadbfd14e3c",
    ),
    (
        "source-image.bin",
        25_050,
        "afb5f0e4f5373c506d104279f3d25b436b3a3cf19bac5ea60eb244d72546bf2c",
    ),
    (
        "stage-checkpoints.safetensors",
        6_024_568,
        "d714cf001c47b8ecabe4d88d970ebb5baa62a85da7731bf83e92c855d313950d",
    ),
    (
        "tensor-stats.jsonl",
        3_079,
        "e50d5c22730d4c9bde800a8cbfc8cbf632f4cee5ad8e97a5416e5fb388137b32",
    ),
    (
        "token-trace.jsonl",
        1_694,
        "876d2ebb68291bfc9a3bd1a39126fcf7159385e424cd4ea0533d71a967ed0bfe",
    ),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisionStackTensorCatalog {
    Model,
    Processor,
    DeepCheckpoints,
    StageCheckpoints,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisionStackInputMaterialization {
    CapturedEmbedding,
    PatchProjectionWithInterpolatedPosition {
        channels: u32,
        patch_size: u32,
        source_height: u32,
        source_width: u32,
        image_grid_thw: [u32; 3],
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionStackTensorSpec {
    pub catalog: VisionStackTensorCatalog,
    pub name: String,
    pub shape: Vec<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionStackShardTensorSpec {
    pub id: String,
    pub kind: VisionStackShardKind,
    pub layer_index: Option<u32>,
    pub tensors: Vec<VisionStackTensorSpec>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OfficialVisionStackTensorProgram {
    pub input_materialization: VisionStackInputMaterialization,
    pub shards: Vec<VisionStackShardTensorSpec>,
    pub expected: Vec<VisionStackTensorSpec>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompileOfficialVisionStackOptions {
    pub compiler_build: String,
    pub profile: OfficialVisionStackProfile,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OfficialVisionStackCompilationReport {
    pub manifest: VisionStackShardManifest,
    pub expected_bytes: u64,
    pub expected_blake3: String,
}

#[must_use]
pub fn official_vision_stack_tensor_program() -> OfficialVisionStackTensorProgram {
    official_vision_stack_tensor_program_for(OfficialVisionStackProfile::OcrCleanLatinL3)
}

#[must_use]
pub fn official_vision_stack_tensor_program_for(
    profile: OfficialVisionStackProfile,
) -> OfficialVisionStackTensorProgram {
    let tokens = profile.tokens();
    let (input_materialization, input_tensors) = match profile {
        OfficialVisionStackProfile::OcrCleanLatinL3 => (
            VisionStackInputMaterialization::CapturedEmbedding,
            vec![tensor(
                VisionStackTensorCatalog::DeepCheckpoints,
                "vision.embeddings.output",
                &[1, u64::from(tokens), u64::from(HIDDEN)],
            )],
        ),
        OfficialVisionStackProfile::TableSimpleL2 => (
            VisionStackInputMaterialization::PatchProjectionWithInterpolatedPosition {
                channels: 3,
                patch_size: 14,
                source_height: 27,
                source_width: 27,
                image_grid_thw: [1, 30, 58],
            },
            vec![
                tensor(
                    VisionStackTensorCatalog::Processor,
                    "processor.pixel_values",
                    &[u64::from(tokens), 3, 14, 14],
                ),
                tensor(
                    VisionStackTensorCatalog::Model,
                    "visual.vision_model.embeddings.patch_embedding.weight",
                    &[u64::from(HIDDEN), 3, 14, 14],
                ),
                tensor(
                    VisionStackTensorCatalog::Model,
                    "visual.vision_model.embeddings.patch_embedding.bias",
                    &[u64::from(HIDDEN)],
                ),
                tensor(
                    VisionStackTensorCatalog::Model,
                    "visual.vision_model.embeddings.position_embedding.weight",
                    &[729, u64::from(HIDDEN)],
                ),
            ],
        ),
    };
    let input = VisionStackShardTensorSpec {
        id: "input.embeddings".to_owned(),
        kind: VisionStackShardKind::Input,
        layer_index: None,
        tensors: input_tensors,
    };
    let mut shards = Vec::with_capacity(LAYERS as usize + 2);
    shards.push(input);
    for layer in 0..LAYERS {
        let prefix = format!("visual.vision_model.encoder.layers.{layer}");
        let named = |suffix: &str, shape: &[u64]| {
            tensor(
                VisionStackTensorCatalog::Model,
                &format!("{prefix}.{suffix}"),
                shape,
            )
        };
        let hidden = u64::from(HIDDEN);
        let intermediate = u64::from(INTERMEDIATE);
        shards.push(VisionStackShardTensorSpec {
            id: format!("weights.vision_layer.{layer:02}"),
            kind: VisionStackShardKind::Layer,
            layer_index: Some(layer),
            tensors: vec![
                named("layer_norm1.weight", &[hidden]),
                named("layer_norm1.bias", &[hidden]),
                named("self_attn.q_proj.weight", &[hidden, hidden]),
                named("self_attn.q_proj.bias", &[hidden]),
                named("self_attn.k_proj.weight", &[hidden, hidden]),
                named("self_attn.k_proj.bias", &[hidden]),
                named("self_attn.v_proj.weight", &[hidden, hidden]),
                named("self_attn.v_proj.bias", &[hidden]),
                named("self_attn.out_proj.weight", &[hidden, hidden]),
                named("self_attn.out_proj.bias", &[hidden]),
                named("layer_norm2.weight", &[hidden]),
                named("layer_norm2.bias", &[hidden]),
                named("mlp.fc1.weight", &[intermediate, hidden]),
                named("mlp.fc1.bias", &[intermediate]),
                named("mlp.fc2.weight", &[hidden, intermediate]),
                named("mlp.fc2.bias", &[hidden]),
            ],
        });
    }
    shards.push(VisionStackShardTensorSpec {
        id: "weights.vision_post_norm".to_owned(),
        kind: VisionStackShardKind::PostNorm,
        layer_index: None,
        tensors: vec![
            tensor(
                VisionStackTensorCatalog::Model,
                "visual.vision_model.post_layernorm.weight",
                &[u64::from(HIDDEN)],
            ),
            tensor(
                VisionStackTensorCatalog::Model,
                "visual.vision_model.post_layernorm.bias",
                &[u64::from(HIDDEN)],
            ),
        ],
    });
    let mut expected = profile
        .checkpoint_layers()
        .iter()
        .copied()
        .map(|layer| {
            tensor(
                VisionStackTensorCatalog::DeepCheckpoints,
                &format!("vision.layer.{layer:02}.output"),
                &[1, u64::from(tokens), u64::from(HIDDEN)],
            )
        })
        .collect::<Vec<_>>();
    expected.push(tensor(
        VisionStackTensorCatalog::StageCheckpoints,
        "vision.final",
        &[u64::from(tokens), u64::from(HIDDEN)],
    ));
    OfficialVisionStackTensorProgram {
        input_materialization,
        shards,
        expected,
    }
}

pub fn compile_official_vision_stack_shards(
    lock_path: impl AsRef<Path>,
    model_dir: impl AsRef<Path>,
    golden_dir: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
    options: &CompileOfficialVisionStackOptions,
) -> Result<OfficialVisionStackCompilationReport, SourceError> {
    if !is_lower_hex_64(&options.compiler_build) {
        return Err(SourceError::new(
            SourceErrorCode::InvalidCompilerBuild,
            "compiler build must be exactly 64 lowercase hexadecimal characters",
        ));
    }
    let lock = ModelLock::from_path(lock_path)?;
    verify_model_source(&lock, &model_dir)?;
    let golden_dir = golden_dir.as_ref();
    verify_pinned_golden(golden_dir, options.profile)?;
    let catalogs = Catalogs {
        model: open_catalog(model_dir.as_ref().join("model.safetensors"))?,
        processor: open_catalog(golden_dir.join("processor.safetensors"))?,
        deep: match options.profile {
            OfficialVisionStackProfile::OcrCleanLatinL3 => Some(open_catalog(
                golden_dir.join("deep-checkpoints.safetensors"),
            )?),
            OfficialVisionStackProfile::TableSimpleL2 => None,
        },
        stage: open_catalog(golden_dir.join("stage-checkpoints.safetensors"))?,
    };
    let program = official_vision_stack_tensor_program_for(options.profile);
    preflight_tensor_program(&catalogs, &program)?;

    let output_dir = output_dir.as_ref();
    prepare_output_directory(output_dir, &program)?;
    let mut descriptors = Vec::with_capacity(program.shards.len());
    for (index, shard) in program.shards.iter().enumerate() {
        let payload = if index == 0 {
            materialize_input(&catalogs, &program)?
        } else {
            materialize_tensors(&catalogs, &shard.tensors)?
        };
        let (expected_bytes, expected_blake3) = official_shard_anchor(options.profile, &shard.id)?;
        verify_payload_anchor(&shard.id, &payload, expected_bytes, expected_blake3)?;
        write_atomic(&output_dir.join(format!("{}.f32", shard.id)), &payload)?;
        descriptors.push(VisionStackShardDescriptor {
            id: shard.id.clone(),
            kind: shard.kind,
            layer_index: shard.layer_index,
            bytes: expected_bytes,
            blake3: expected_blake3.to_owned(),
        });
    }
    let expected = materialize_tensors(&catalogs, &program.expected)?;
    let (expected_checkpoint_bytes, expected_checkpoint_blake3) =
        expected_checkpoint_anchor(options.profile);
    verify_payload_anchor(
        "expected.checkpoints",
        &expected,
        expected_checkpoint_bytes,
        expected_checkpoint_blake3,
    )?;
    write_atomic(&output_dir.join("expected.checkpoints.f32"), &expected)?;

    let manifest = VisionStackShardManifest {
        schema_version: VISION_STACK_SHARD_MANIFEST_SCHEMA_VERSION,
        oracle: VisionStackShardOracle::OfficialMpsBf16,
        case_id: options.profile.case_id().to_owned(),
        model_id: MODEL_ID.to_owned(),
        model_revision: MODEL_REVISION.to_owned(),
        compiler_model_abi: COMPILER_MODEL_ABI,
        compiler_build: options.compiler_build.clone(),
        golden_bundle_digest: Some(options.profile.golden_bundle_digest().to_owned()),
        semantic_fingerprint: Some(options.profile.semantic_fingerprint().to_owned()),
        matrix_weight_storage: DecoderWeightStorage::F32,
        matrix_weight_layout: pvlc_pack::LinearWeightLayout::OutputMajor,
        vector_weight_storage: DecoderWeightStorage::F32,
        activation_storage: DecoderWeightStorage::F32,
        tokens: options.profile.tokens(),
        hidden_size: HIDDEN,
        attention_heads: HEADS,
        head_dim: HEAD_DIM,
        intermediate_size: INTERMEDIATE,
        layer_norm_epsilon: LAYER_NORM_EPSILON,
        cu_seqlens: vec![0, options.profile.tokens()],
        layer_count: LAYERS,
        checkpoint_layers: options.profile.checkpoint_layers().to_vec(),
        shards: descriptors,
    };
    let manifest_bytes =
        canonical_vision_stack_shard_manifest_bytes(&manifest).map_err(vision_stack_error)?;
    // The canonical manifest is the commit record and is deliberately installed last.
    write_atomic(&output_dir.join("manifest.json"), &manifest_bytes)?;
    Ok(OfficialVisionStackCompilationReport {
        manifest,
        expected_bytes: expected_checkpoint_bytes,
        expected_blake3: expected_checkpoint_blake3.to_owned(),
    })
}

fn tensor(catalog: VisionStackTensorCatalog, name: &str, shape: &[u64]) -> VisionStackTensorSpec {
    VisionStackTensorSpec {
        catalog,
        name: name.to_owned(),
        shape: shape.to_vec(),
    }
}

struct Catalogs {
    model: SafetensorsCatalog,
    processor: SafetensorsCatalog,
    deep: Option<SafetensorsCatalog>,
    stage: SafetensorsCatalog,
}

impl Catalogs {
    fn get(&self, catalog: VisionStackTensorCatalog) -> Result<&SafetensorsCatalog, SourceError> {
        match catalog {
            VisionStackTensorCatalog::Model => Ok(&self.model),
            VisionStackTensorCatalog::Processor => Ok(&self.processor),
            VisionStackTensorCatalog::DeepCheckpoints => self.deep.as_ref().ok_or_else(|| {
                SourceError::new(
                    SourceErrorCode::Safetensors,
                    "selected vision-stack profile has no deep-checkpoint catalog",
                )
            }),
            VisionStackTensorCatalog::StageCheckpoints => Ok(&self.stage),
        }
    }
}

fn open_catalog(path: impl AsRef<Path>) -> Result<SafetensorsCatalog, SourceError> {
    SafetensorsCatalog::open(path).map_err(safetensors_error)
}

fn preflight_tensor_program(
    catalogs: &Catalogs,
    program: &OfficialVisionStackTensorProgram,
) -> Result<(), SourceError> {
    for spec in program
        .shards
        .iter()
        .flat_map(|shard| &shard.tensors)
        .chain(&program.expected)
    {
        let header = catalogs
            .get(spec.catalog)?
            .tensor(&spec.name)
            .ok_or_else(|| {
                SourceError::at_path(
                    SourceErrorCode::Safetensors,
                    &spec.name,
                    "required compiler tensor is missing",
                )
            })?;
        if header.shape != spec.shape {
            return Err(SourceError::at_path(
                SourceErrorCode::Safetensors,
                &spec.name,
                format!(
                    "required compiler tensor has shape {:?}, expected {:?}",
                    header.shape, spec.shape
                ),
            ));
        }
        if !matches!(header.dtype, TensorDtype::BFloat16 | TensorDtype::Float32) {
            return Err(SourceError::at_path(
                SourceErrorCode::Safetensors,
                &spec.name,
                "required compiler tensor is not BF16 or F32",
            ));
        }
    }
    Ok(())
}

fn materialize_input(
    catalogs: &Catalogs,
    program: &OfficialVisionStackTensorProgram,
) -> Result<Vec<u8>, SourceError> {
    let input = program.shards.first().ok_or_else(|| {
        SourceError::new(
            SourceErrorCode::VisionStackShard,
            "vision-stack tensor program has no input shard",
        )
    })?;
    match program.input_materialization {
        VisionStackInputMaterialization::CapturedEmbedding => {
            materialize_tensors(catalogs, &input.tensors)
        }
        VisionStackInputMaterialization::PatchProjectionWithInterpolatedPosition {
            channels,
            patch_size,
            source_height,
            source_width,
            image_grid_thw,
        } => {
            let [pixels, weights, bias, positions] = input.tensors.as_slice() else {
                return Err(SourceError::new(
                    SourceErrorCode::VisionStackShard,
                    "patch-plus-position input recipe must have exactly four tensors",
                ));
            };
            let pixels = load_finite_tensor(catalogs, pixels)?;
            let weights = load_finite_tensor(catalogs, weights)?;
            let bias = load_finite_tensor(catalogs, bias)?;
            let positions = load_finite_tensor(catalogs, positions)?;
            let tokens = image_grid_thw
                .into_iter()
                .try_fold(1_u32, |total, dimension| total.checked_mul(dimension))
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    SourceError::new(
                        SourceErrorCode::VisionStackShard,
                        "patch-plus-position input token count overflowed",
                    )
                })?;
            let patches = patch_projection_f32(
                &pixels,
                tokens,
                usize::try_from(channels).map_err(|_| input_recipe_overflow())?,
                usize::try_from(patch_size).map_err(|_| input_recipe_overflow())?,
                &weights,
                &bias,
                usize::try_from(HIDDEN).expect("fixed hidden size fits usize"),
            )
            .map_err(cpu_ref_error)?;
            let grid = [image_grid_thw.map(|value| {
                usize::try_from(value).expect("reviewed image-grid dimensions fit usize")
            })];
            let embedding = add_interpolated_position_embedding_f32(
                &patches,
                usize::try_from(HIDDEN).expect("fixed hidden size fits usize"),
                &positions,
                usize::try_from(source_height).map_err(|_| input_recipe_overflow())?,
                usize::try_from(source_width).map_err(|_| input_recipe_overflow())?,
                &grid,
            )
            .map_err(cpu_ref_error)?;
            f32_values_to_le_bytes(&embedding, "materialized vision-stack input")
        }
    }
}

fn materialize_tensors(
    catalogs: &Catalogs,
    tensors: &[VisionStackTensorSpec],
) -> Result<Vec<u8>, SourceError> {
    let capacity = tensors
        .iter()
        .try_fold(0_usize, |total, tensor| {
            tensor
                .shape
                .iter()
                .try_fold(1_u64, |elements, dimension| {
                    elements.checked_mul(*dimension)
                })
                .and_then(|elements| elements.checked_mul(4))
                .and_then(|bytes| usize::try_from(bytes).ok())
                .and_then(|bytes| total.checked_add(bytes))
        })
        .ok_or_else(|| {
            SourceError::new(
                SourceErrorCode::Safetensors,
                "vision-stack tensor payload size overflowed",
            )
        })?;
    let mut payload = Vec::with_capacity(capacity);
    for tensor in tensors {
        let values = load_finite_tensor(catalogs, tensor)?;
        for value in values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
    }
    if payload.len() != capacity {
        return Err(SourceError::new(
            SourceErrorCode::Safetensors,
            "materialized tensor payload length drifted from its preflight shapes",
        ));
    }
    Ok(payload)
}

fn load_finite_tensor(
    catalogs: &Catalogs,
    tensor: &VisionStackTensorSpec,
) -> Result<Vec<f32>, SourceError> {
    let values = catalogs
        .get(tensor.catalog)?
        .load_tensor_f32(&tensor.name)
        .map_err(safetensors_error)?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(SourceError::at_path(
            SourceErrorCode::Safetensors,
            &tensor.name,
            "required compiler tensor contains a non-finite value",
        ));
    }
    Ok(values)
}

fn f32_values_to_le_bytes(values: &[f32], label: &str) -> Result<Vec<u8>, SourceError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(SourceError::at_path(
            SourceErrorCode::VisionStackShard,
            label,
            "materialized values contain a non-finite value",
        ));
    }
    Ok(values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect())
}

fn input_recipe_overflow() -> SourceError {
    SourceError::new(
        SourceErrorCode::VisionStackShard,
        "patch-plus-position input recipe does not fit this compiler target",
    )
}

fn cpu_ref_error(error: pvlc_cpu_ref::CpuRefError) -> SourceError {
    SourceError::new(SourceErrorCode::VisionStackShard, error.to_string())
}

fn official_shard_anchor(
    profile: OfficialVisionStackProfile,
    id: &str,
) -> Result<(u64, &'static str), SourceError> {
    if profile == OfficialVisionStackProfile::TableSimpleL2 && id == "input.embeddings" {
        return Ok((TABLE_L2_INPUT_BYTES, TABLE_L2_INPUT_BLAKE3));
    }
    OFFICIAL_SHARD_ANCHORS
        .iter()
        .find(|(candidate, _, _)| *candidate == id)
        .map(|(_, bytes, digest)| (*bytes, *digest))
        .ok_or_else(|| {
            SourceError::at_path(
                SourceErrorCode::VisionStackShard,
                id,
                "official shard has no independently reviewed anchor",
            )
        })
}

fn expected_checkpoint_anchor(profile: OfficialVisionStackProfile) -> (u64, &'static str) {
    match profile {
        OfficialVisionStackProfile::OcrCleanLatinL3 => {
            (EXPECTED_CHECKPOINT_BYTES, EXPECTED_CHECKPOINT_BLAKE3)
        }
        OfficialVisionStackProfile::TableSimpleL2 => {
            (TABLE_L2_EXPECTED_BYTES, TABLE_L2_EXPECTED_BLAKE3)
        }
    }
}

fn verify_payload_anchor(
    id: &str,
    payload: &[u8],
    expected_bytes: u64,
    expected_blake3: &str,
) -> Result<(), SourceError> {
    let digest = blake3::hash(payload);
    if payload.len() as u64 != expected_bytes || digest.to_hex().as_str() != expected_blake3 {
        return Err(SourceError::at_path(
            SourceErrorCode::VisionStackShard,
            id,
            format!(
                "compiled payload is {} bytes / {}, expected {expected_bytes} bytes / {expected_blake3}",
                payload.len(),
                digest.to_hex()
            ),
        ));
    }
    Ok(())
}

fn prepare_output_directory(
    output: &Path,
    program: &OfficialVisionStackTensorProgram,
) -> Result<(), SourceError> {
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let parent_metadata = fs::symlink_metadata(parent).map_err(|error| {
        SourceError::io(
            error,
            format!("cannot inspect compiler output parent {}", parent.display()),
        )
    })?;
    if !parent_metadata.file_type().is_dir() {
        return Err(SourceError::new(
            SourceErrorCode::InvalidOutput,
            format!(
                "compiler output parent {} is not a real directory",
                parent.display()
            ),
        ));
    }
    match fs::symlink_metadata(output) {
        Ok(metadata) if !metadata.file_type().is_dir() => {
            return Err(SourceError::new(
                SourceErrorCode::InvalidOutput,
                format!(
                    "compiler output {} is not a real directory",
                    output.display()
                ),
            ));
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir(output).map_err(|error| {
                SourceError::io(
                    error,
                    format!("cannot create compiler output {}", output.display()),
                )
            })?;
        }
        Err(error) => {
            return Err(SourceError::io(
                error,
                format!("cannot inspect compiler output {}", output.display()),
            ));
        }
    }
    let mut allowed = program
        .shards
        .iter()
        .map(|shard| format!("{}.f32", shard.id))
        .collect::<BTreeSet<_>>();
    allowed.insert("expected.checkpoints.f32".to_owned());
    allowed.insert("manifest.json".to_owned());
    for entry in fs::read_dir(output).map_err(|error| {
        SourceError::io(
            error,
            format!("cannot enumerate compiler output {}", output.display()),
        )
    })? {
        let entry = entry.map_err(|error| SourceError::io(error, "cannot read output entry"))?;
        let name = entry.file_name().into_string().map_err(|_| {
            SourceError::new(
                SourceErrorCode::InvalidOutput,
                "output entry name is not UTF-8",
            )
        })?;
        let file_type = entry
            .file_type()
            .map_err(|error| SourceError::io(error, "cannot inspect output entry"))?;
        if !file_type.is_file() || !allowed.contains(&name) {
            return Err(SourceError::at_path(
                SourceErrorCode::InvalidOutput,
                name,
                "compiler output contains an unexpected or non-regular entry",
            ));
        }
    }
    Ok(())
}

fn verify_pinned_golden(
    root: &Path,
    profile: OfficialVisionStackProfile,
) -> Result<(), SourceError> {
    let metadata = fs::symlink_metadata(root).map_err(|error| {
        SourceError::io(
            error,
            format!("cannot inspect golden directory {}", root.display()),
        )
    })?;
    if !metadata.file_type().is_dir() {
        return Err(SourceError::new(
            SourceErrorCode::GoldenMismatch,
            "official golden source is not a real directory",
        ));
    }
    let pinned_files: &[(&str, u64, &str)] = match profile {
        OfficialVisionStackProfile::OcrCleanLatinL3 => &PINNED_GOLDEN_FILES,
        OfficialVisionStackProfile::TableSimpleL2 => &PINNED_TABLE_L2_GOLDEN_FILES,
    };
    let expected_names = pinned_files
        .iter()
        .map(|(name, _, _)| (*name).to_owned())
        .collect::<BTreeSet<_>>();
    let mut observed_names = BTreeSet::new();
    for entry in fs::read_dir(root).map_err(|error| {
        SourceError::io(
            error,
            format!("cannot enumerate golden directory {}", root.display()),
        )
    })? {
        let entry = entry.map_err(|error| SourceError::io(error, "cannot read golden entry"))?;
        let name = entry.file_name().into_string().map_err(|_| {
            SourceError::new(
                SourceErrorCode::GoldenMismatch,
                "golden entry name is not UTF-8",
            )
        })?;
        let file_type = entry
            .file_type()
            .map_err(|error| SourceError::io(error, "cannot inspect golden entry"))?;
        if !file_type.is_file() {
            return Err(SourceError::at_path(
                SourceErrorCode::GoldenMismatch,
                name,
                "golden bundle contains a symlink, directory, or special file",
            ));
        }
        observed_names.insert(name);
    }
    if observed_names != expected_names {
        return Err(SourceError::new(
            SourceErrorCode::GoldenMismatch,
            "golden bundle inventory differs from the selected pinned profile",
        ));
    }
    for &(name, bytes, digest) in pinned_files {
        verify_regular_file(&root.join(name), name, bytes, digest)?;
    }
    Ok(())
}

fn verify_regular_file(
    path: &Path,
    label: &str,
    expected_bytes: u64,
    expected_blake3: &str,
) -> Result<(), SourceError> {
    let before = fs::symlink_metadata(path).map_err(|error| {
        SourceError::io(
            error,
            format!("cannot inspect pinned file {}", path.display()),
        )
    })?;
    if !before.file_type().is_file() || before.len() != expected_bytes {
        return Err(SourceError::at_path(
            SourceErrorCode::GoldenMismatch,
            label,
            "pinned file type or byte length drifted",
        ));
    }
    let mut file = File::open(path).map_err(|error| {
        SourceError::io(error, format!("cannot open pinned file {}", path.display()))
    })?;
    let mut hasher = blake3::Hasher::new();
    let mut total = 0_u64;
    let mut buffer = [0_u8; HASH_BUFFER_BYTES];
    loop {
        let read = file.read(&mut buffer).map_err(|error| {
            SourceError::io(error, format!("cannot hash pinned file {}", path.display()))
        })?;
        if read == 0 {
            break;
        }
        total = total.checked_add(read as u64).ok_or_else(|| {
            SourceError::at_path(
                SourceErrorCode::GoldenMismatch,
                label,
                "pinned file byte count overflowed",
            )
        })?;
        hasher.update(&buffer[..read]);
    }
    let after = file.metadata().map_err(|error| {
        SourceError::io(
            error,
            format!("cannot restat pinned file {}", path.display()),
        )
    })?;
    let digest = hasher.finalize();
    if total != expected_bytes
        || !same_snapshot(&before, &after)
        || digest.to_hex().as_str() != expected_blake3
    {
        return Err(SourceError::at_path(
            SourceErrorCode::GoldenMismatch,
            label,
            "pinned file changed or its BLAKE3 differs from the reviewed bundle",
        ));
    }
    Ok(())
}

fn safetensors_error(error: pvlc_safetensors::ImportError) -> SourceError {
    SourceError::new(SourceErrorCode::Safetensors, error.to_string())
}

fn vision_stack_error(error: pvlc_pack::VisionStackShardError) -> SourceError {
    SourceError::new(SourceErrorCode::VisionStackShard, error.to_string())
}
