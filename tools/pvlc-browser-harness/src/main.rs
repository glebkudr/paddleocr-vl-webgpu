use std::{
    collections::BTreeMap,
    env,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use clap::{Parser, Subcommand};
use pvlc_cpu_ref::{
    KvBlockOrder, LayerNormParameters as CpuLayerNormParameters,
    LinearParameters as CpuLinearParameters, ProjectorParameters as CpuProjectorParameters,
    ProjectorTrace as CpuProjectorTrace, VisionEncoderLayerConfig as CpuVisionLayerConfig,
    VisionEncoderLayerParameters as CpuVisionLayerParameters, VisionEncoderStackConfig,
    projector_f32 as cpu_projector_f32, vision_encoder_layer_identity_rope_f32,
    vision_encoder_stack_identity_rope_f32,
};
use pvlc_model_schema::{COMPILER_MODEL_ABI, MODEL_ID, MODEL_REVISION as PINNED_MODEL_REVISION};
use pvlc_pack::{
    ProjectorSelfTestCaseSource, ProjectorSelfTestSource,
    VISION_STACK_SHARD_MANIFEST_SCHEMA_VERSION, VisionLayerSelfTestSource,
    VisionStackShardDescriptor, VisionStackShardKind, VisionStackShardManifest,
    VisionStackShardOracle, build_projector_self_test_pack, build_vision_layer_self_test_pack,
    canonical_vision_stack_shard_manifest_bytes,
};
use pvlc_runtime_core::{
    KernelId, KernelInvocation, OwnedProjectorInvocation, OwnedProjectorParameters,
    OwnedVisionEncoderLayerInvocation, OwnedVisionEncoderLayerParameters,
    OwnedVisionLayerNormParameters, OwnedVisionLinearParameters, ProjectorReadback, ProjectorStage,
    VisionEncoderLayerStage,
};
use pvlc_runtime_native::{
    BackendKind, ErrorScopeKind, GpuTimestamp, KernelDiagnostics, NativeOptions, NativeRuntime,
    ProjectorDiagnostics, VisionLayerDiagnostics, VisionLayerReadback,
};
use pvlc_safetensors::SafetensorsCatalog;
use pvlc_testkit::{
    ComparisonAxes, ComparisonPolicy, M2PrimitiveCase, M2PrimitiveCorpus, M3VisionAttentionCase,
    M3VisionAttentionCorpus, M3VisionLayerCase, M3VisionLayerCorpus, compare_f32,
    m2_primitive_corpus, m3_vision_attention_corpus, m3_vision_layer_corpus,
};
use serde::Serialize;
use tempfile::NamedTempFile;
use thiserror::Error;

const REQUIRED_GATES: [&str; 3] = [
    "PVLC_REQUIRE_NATIVE_GPU",
    "PVLC_REQUIRE_M4_METAL",
    "PVLC_REQUIRE_TIMESTAMP_QUERY",
];
const MODEL_REVISION: &str = "66317acc4c9fc17bd154591ce650735cd2855f3e";
const OFFICIAL_COMPILER_BUILD: &str =
    "4e40a34cd64f347a14270d339ab61b131ae3120ca566ff140c81d101a3aa4a2c";
const PROJECTOR_ORACLE: &str = "pvlc-cpu-ref/projector-f32-v1";
const PROJECTOR_HIDDEN: usize = 3;
const PROJECTOR_MERGED: usize = PROJECTOR_HIDDEN * 4;
const PROJECTOR_OUTPUT: usize = 5;
const PROJECTOR_EPSILON: f32 = 1.0e-5;
const PROJECTOR_GRIDS: [[u32; 3]; 2] = [[1, 2, 4], [2, 2, 2]];
const PROJECTOR_POLICY: ProjectorPolicy = ProjectorPolicy {
    max_abs: 2.0e-4,
    max_mean_abs: 3.0e-5,
    max_p99_abs: 1.0e-4,
    max_relative_l2: 1.0e-4,
    min_cosine_similarity: 0.99999,
    native_max_abs: 2.0e-4,
    native_max_relative_l2: 1.0e-4,
};

#[derive(Parser)]
#[command(name = "pvlc-browser-harness")]
#[command(about = "Generate fail-closed native/browser comparison artifacts")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Prepare {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        native_baseline: PathBuf,
    },
    PrepareVisionAttention {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        native_baseline: PathBuf,
    },
    PrepareVisionLayer {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        native_baseline: PathBuf,
    },
    PrepareProjector {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        native_baseline: PathBuf,
    },
    PrepareOfficialVisionLayer {
        #[arg(long)]
        pack: PathBuf,
    },
    PrepareOfficialProjector {
        #[arg(long)]
        pack: PathBuf,
    },
    PrepareVisionStackSharded {
        #[arg(long)]
        output_dir: PathBuf,
    },
}

#[derive(Debug, Error)]
enum HarnessError {
    #[error("hardware gate {0}=1 is mandatory for a browser baseline")]
    MissingGate(&'static str),
    #[error("M2 corpus generation failed: {0}")]
    Corpus(#[from] pvlc_testkit::M2CorpusError),
    #[error("M3 vision attention corpus generation failed: {0}")]
    VisionCorpus(#[from] pvlc_testkit::M3VisionAttentionCorpusError),
    #[error("M3 vision-layer corpus generation failed: {0}")]
    VisionLayerCorpus(#[from] pvlc_testkit::M3VisionLayerCorpusError),
    #[error("compact projector CPU oracle failed: {0}")]
    ProjectorCpu(#[source] pvlc_cpu_ref::CpuRefError),
    #[error("compact vision-stack CPU oracle failed: {0}")]
    VisionStackCpu(#[from] pvlc_cpu_ref::CpuRefError),
    #[error("compact vision-stack manifest generation failed: {0}")]
    VisionStackManifest(#[from] pvlc_pack::VisionStackShardError),
    #[error("official safetensors loading failed: {0}")]
    Safetensors(#[from] pvlc_safetensors::ImportError),
    #[error("official vision-layer pack generation failed: {0}")]
    VisionLayerPack(#[from] pvlc_pack::VisionLayerSelfTestError),
    #[error("official projector pack generation failed: {0}")]
    ProjectorPack(#[from] pvlc_pack::ProjectorSelfTestError),
    #[error("official tensor {name} has shape {actual:?}, expected {expected:?}")]
    OfficialTensorShape {
        name: String,
        actual: Vec<u64>,
        expected: Vec<u64>,
    },
    #[error("official tensor {0} is missing")]
    MissingOfficialTensor(String),
    #[error("native runtime initialization failed: {0}")]
    NativeRuntime(#[from] pvlc_runtime_native::RuntimeError),
    #[error("native baseline requires Apple M4 Pro Metal, got {backend:?} / {adapter}")]
    WrongAdapter {
        backend: BackendKind,
        adapter: String,
    },
    #[error("native baseline requires timestamp-query support")]
    TimestampUnavailable,
    #[error("native execution for {case_id} returned invalid evidence: {message}")]
    InvalidEvidence { case_id: String, message: String },
    #[error("native execution for {case_id} diverged from CPU oracle: {message}")]
    CpuMismatch { case_id: String, message: String },
    #[error("cannot create output directory {path}: {source}")]
    CreateDirectory {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("cannot create temporary output next to {path}: {source}")]
    CreateTemporary {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("cannot serialize {path}: {source}")]
    Serialize {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("cannot flush {path}: {source}")]
    Flush {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("cannot persist {path}: {source}")]
    Persist {
        path: PathBuf,
        source: tempfile::PersistError,
    },
    #[error("output path {0} has no parent directory")]
    MissingParent(PathBuf),
}

#[derive(Serialize)]
struct CorpusArtifact<'a> {
    schema_version: u32,
    oracle: &'a str,
    corpus_blake3: &'a str,
    cases: &'a [M2PrimitiveCase],
}

#[derive(Serialize)]
struct VisionCorpusArtifact<'a> {
    schema_version: u32,
    oracle: &'a str,
    fixture_algorithm: &'a str,
    corpus_blake3: &'a str,
    cases: &'a [M3VisionAttentionCase],
}

#[derive(Serialize)]
struct VisionLayerCorpusArtifact<'a> {
    schema_version: u32,
    oracle: &'a str,
    fixture_algorithm: &'a str,
    corpus_blake3: &'a str,
    cases: &'a [M3VisionLayerCase],
}

#[derive(Serialize)]
struct NativeBaseline {
    schema_version: u32,
    corpus_blake3: String,
    capabilities: NativeCapabilitiesReport,
    submission_count: u64,
    cases: Vec<NativeCase>,
}

#[derive(Serialize)]
struct NativeCapabilitiesReport {
    adapter_name: String,
    backend: &'static str,
    timestamp_query: bool,
    max_storage_buffer_binding_size: u64,
    max_compute_workgroups_per_dimension: u32,
    max_compute_invocations_per_workgroup: u32,
    max_compute_workgroup_size_x: u32,
    max_compute_workgroup_size_y: u32,
    max_compute_workgroup_size_z: u32,
    max_compute_workgroup_storage_size: u32,
    max_storage_buffers_per_shader_stage: u32,
    max_buffer_size: u64,
}

#[derive(Serialize)]
struct NativeCase {
    id: String,
    kernel: KernelId,
    values: Vec<f32>,
    diagnostics: NativeDiagnosticsReport,
}

#[derive(Serialize)]
struct NativeDiagnosticsReport {
    checked_error_scopes: [&'static str; 3],
    captured_errors: Vec<String>,
    queue_wall_time_ns: u64,
    shader_blake3: String,
    timestamp: NativeTimestampReport,
}

#[derive(Serialize)]
struct NativeTimestampReport {
    begin_ticks: u64,
    end_ticks: u64,
    duration_ns: f64,
}

#[derive(Serialize)]
struct VisionLayerNativeBaseline {
    schema_version: u32,
    corpus_blake3: String,
    capabilities: NativeCapabilitiesReport,
    submission_count: u64,
    cases: Vec<VisionLayerNativeCase>,
}

#[derive(Serialize)]
struct VisionLayerNativeCase {
    id: String,
    checkpoints: BTreeMap<VisionEncoderLayerStage, Vec<f32>>,
    diagnostics: VisionLayerNativeDiagnosticsReport,
}

#[derive(Serialize)]
struct VisionLayerNativeDiagnosticsReport {
    checked_error_scopes: [&'static str; 3],
    captured_errors: Vec<String>,
    queue_wall_time_ns: u64,
    shader_blake3: BTreeMap<KernelId, String>,
    dispatch_stages: [VisionEncoderLayerStage; 12],
    rope_specialization: pvlc_runtime_core::VisionRopeSpecialization,
    submission_count: u64,
    command_buffer_count: u32,
    compute_pass_count: u32,
    dispatch_count: u32,
    buffer_allocation_count: u64,
    readback_buffer_count: u32,
    readback_bytes: u64,
    timestamp: NativeTimestampReport,
}

#[derive(Clone, Copy, Serialize)]
struct ProjectorPolicy {
    max_abs: f64,
    max_mean_abs: f64,
    max_p99_abs: f64,
    max_relative_l2: f64,
    min_cosine_similarity: f64,
    native_max_abs: f64,
    native_max_relative_l2: f64,
}

impl ProjectorPolicy {
    fn comparison_policy(self) -> ComparisonPolicy {
        ComparisonPolicy {
            require_finite: true,
            max_abs: self.max_abs,
            max_mean_abs: self.max_mean_abs,
            max_p99_abs: self.max_p99_abs,
            max_relative_l2: self.max_relative_l2,
            min_cosine_similarity: self.min_cosine_similarity,
            max_per_token_relative_l2: None,
            max_per_channel_relative_l2: None,
        }
    }

    fn native_comparison_policy(self) -> ComparisonPolicy {
        ComparisonPolicy {
            max_abs: self.native_max_abs,
            max_mean_abs: self.native_max_abs,
            max_p99_abs: self.native_max_abs,
            max_relative_l2: self.native_max_relative_l2,
            ..self.comparison_policy()
        }
    }
}

#[derive(Serialize)]
struct ProjectorCase {
    id: String,
    poisoned_image: Option<usize>,
    invocation: OwnedProjectorInvocation,
    expected: BTreeMap<ProjectorStage, Vec<f32>>,
    policy: ProjectorPolicy,
}

#[derive(Serialize)]
struct ProjectorCorpusSource<'a> {
    schema_version: u32,
    oracle: &'a str,
    cases: &'a [ProjectorCase],
}

#[derive(Serialize)]
struct ProjectorCorpusArtifact<'a> {
    schema_version: u32,
    oracle: &'a str,
    corpus_blake3: &'a str,
    cases: &'a [ProjectorCase],
}

#[derive(Serialize)]
struct ProjectorNativeBaseline {
    schema_version: u32,
    corpus_blake3: String,
    capabilities: NativeCapabilitiesReport,
    submission_count: u64,
    cases: Vec<ProjectorNativeCase>,
}

#[derive(Serialize)]
struct ProjectorNativeCase {
    id: String,
    checkpoints: BTreeMap<ProjectorStage, Vec<f32>>,
    diagnostics: ProjectorNativeDiagnosticsReport,
}

#[derive(Serialize)]
struct ProjectorNativeDiagnosticsReport {
    checked_error_scopes: [&'static str; 3],
    captured_errors: Vec<String>,
    queue_wall_time_ns: u64,
    timestamp: ProjectorNativeTimestampReport,
    timestamp_fresh: bool,
    shader_blake3: BTreeMap<KernelId, String>,
    dispatch_stages: [ProjectorStage; 5],
    submission_count: u64,
    command_buffer_count: u32,
    compute_pass_count: u32,
    dispatch_count: u32,
    buffer_allocation_count: u64,
    readback_buffer_count: u32,
    readback_map_count: u32,
    readback_bytes: u64,
    resident_intermediate_bytes: u64,
    resident_weight_bytes: u64,
}

#[derive(Serialize)]
struct ProjectorNativeTimestampReport {
    begin_ticks: u64,
    end_ticks: u64,
    period_ns: f64,
    duration_ns: f64,
}

fn main() -> Result<(), HarnessError> {
    match Cli::parse().command {
        Command::Prepare {
            corpus,
            native_baseline,
        } => prepare(&corpus, &native_baseline),
        Command::PrepareVisionAttention {
            corpus,
            native_baseline,
        } => prepare_vision_attention(&corpus, &native_baseline),
        Command::PrepareVisionLayer {
            corpus,
            native_baseline,
        } => prepare_vision_layer(&corpus, &native_baseline),
        Command::PrepareProjector {
            corpus,
            native_baseline,
        } => prepare_projector(&corpus, &native_baseline),
        Command::PrepareOfficialVisionLayer { pack } => prepare_official_vision_layer(&pack),
        Command::PrepareOfficialProjector { pack } => prepare_official_projector(&pack),
        Command::PrepareVisionStackSharded { output_dir } => {
            prepare_vision_stack_sharded(&output_dir)
        }
    }
}

fn prepare_vision_stack_sharded(output_dir: &Path) -> Result<(), HarnessError> {
    const TOKENS: u32 = 3;
    const HIDDEN: u32 = 4;
    const HEADS: u32 = 2;
    const HEAD_DIM: u32 = 2;
    const INTERMEDIATE: u32 = 5;
    const LAYERS: u32 = 3;
    const EPSILON: f32 = 1.0e-6;
    const BOUNDARIES: [u32; 3] = [0, 1, 3];
    const CHECKPOINTS: [u32; 2] = [0, 2];

    let input = compact_values((TOKENS * HIDDEN) as usize, 0, 3, 16.0, 0.0);
    let layers = (0..LAYERS)
        .map(compact_vision_layer_parameters)
        .collect::<Vec<_>>();
    let post_norm = OwnedVisionLayerNormParameters {
        weight: compact_values(HIDDEN as usize, LAYERS, 89, 256.0, 1.0),
        bias: compact_values(HIDDEN as usize, LAYERS, 97, 256.0, 0.0),
    };
    let boundaries = BOUNDARIES
        .iter()
        .map(|value| *value as usize)
        .collect::<Vec<_>>();
    let trace = vision_encoder_stack_identity_rope_f32(
        &input,
        VisionEncoderStackConfig {
            tokens: TOKENS as usize,
            hidden_size: HIDDEN as usize,
            layers: LAYERS as usize,
            layer_norm_epsilon: EPSILON,
        },
        &CHECKPOINTS.map(|value| value as usize),
        CpuLayerNormParameters {
            weight: &post_norm.weight,
            bias: &post_norm.bias,
        },
        |layer, current| {
            let parameters = compact_cpu_parameters(&layers[layer]);
            vision_encoder_layer_identity_rope_f32(
                current,
                CpuVisionLayerConfig {
                    tokens: TOKENS as usize,
                    hidden_size: HIDDEN as usize,
                    attention_heads: HEADS as usize,
                    head_dim: HEAD_DIM as usize,
                    intermediate_size: INTERMEDIATE as usize,
                    layer_norm_epsilon: EPSILON,
                    attention_key_tile: 4,
                    attention_order: KvBlockOrder::Forward,
                },
                &boundaries,
                parameters,
            )
            .map(|layer_trace| layer_trace.output)
        },
    )?;

    let mut payloads = Vec::with_capacity(LAYERS as usize + 2);
    payloads.push((
        "input.embeddings".to_owned(),
        VisionStackShardKind::Input,
        None,
        f32_le_bytes(&input),
    ));
    for (layer, parameters) in layers.iter().enumerate() {
        payloads.push((
            format!("weights.vision_layer.{layer:02}"),
            VisionStackShardKind::Layer,
            Some(layer as u32),
            vision_layer_parameter_bytes(parameters),
        ));
    }
    let mut post_norm_bytes = f32_le_bytes(&post_norm.weight);
    post_norm_bytes.extend_from_slice(&f32_le_bytes(&post_norm.bias));
    payloads.push((
        "weights.vision_post_norm".to_owned(),
        VisionStackShardKind::PostNorm,
        None,
        post_norm_bytes,
    ));

    let shards = payloads
        .iter()
        .map(
            |(id, kind, layer_index, bytes)| VisionStackShardDescriptor {
                id: id.clone(),
                kind: *kind,
                layer_index: *layer_index,
                bytes: bytes.len() as u64,
                blake3: blake3::hash(bytes).to_hex().to_string(),
            },
        )
        .collect();
    let manifest = VisionStackShardManifest {
        schema_version: VISION_STACK_SHARD_MANIFEST_SCHEMA_VERSION,
        oracle: VisionStackShardOracle::Synthetic,
        case_id: "synthetic.vision_stack/bounded_streaming".to_owned(),
        model_id: MODEL_ID.to_owned(),
        model_revision: PINNED_MODEL_REVISION.to_owned(),
        compiler_model_abi: COMPILER_MODEL_ABI,
        compiler_build: OFFICIAL_COMPILER_BUILD.to_owned(),
        golden_bundle_digest: None,
        semantic_fingerprint: None,
        matrix_weight_storage: pvlc_runtime_core::DecoderWeightStorage::F32,
        matrix_weight_layout: pvlc_runtime_core::LinearWeightLayout::OutputMajor,
        vector_weight_storage: pvlc_runtime_core::DecoderWeightStorage::F32,
        activation_storage: pvlc_runtime_core::DecoderWeightStorage::F32,
        tokens: TOKENS,
        hidden_size: HIDDEN,
        attention_heads: HEADS,
        head_dim: HEAD_DIM,
        intermediate_size: INTERMEDIATE,
        layer_norm_epsilon: EPSILON,
        cu_seqlens: BOUNDARIES.to_vec(),
        layer_count: LAYERS,
        checkpoint_layers: CHECKPOINTS.to_vec(),
        shards,
    };
    manifest.plan()?;

    for (id, _, _, bytes) in &payloads {
        write_bytes_atomic(&output_dir.join(format!("{id}.f32")), bytes)?;
    }
    let mut expected = Vec::with_capacity(trace.retained_checkpoint_elements + trace.output.len());
    for checkpoint in &trace.checkpoints {
        expected.extend_from_slice(&checkpoint.values);
    }
    expected.extend_from_slice(&trace.output);
    write_bytes_atomic(
        &output_dir.join("expected.checkpoints.f32"),
        &f32_le_bytes(&expected),
    )?;
    write_bytes_atomic(
        &output_dir.join("manifest.json"),
        &canonical_vision_stack_shard_manifest_bytes(&manifest)?,
    )
}

fn compact_vision_layer_parameters(layer: u32) -> OwnedVisionEncoderLayerParameters {
    const HIDDEN: usize = 4;
    const INTERMEDIATE: usize = 5;
    let norm = |salt, shift| OwnedVisionLayerNormParameters {
        weight: compact_values(HIDDEN, layer, salt, 256.0, shift),
        bias: compact_values(HIDDEN, layer, salt + 4, 256.0, 0.0),
    };
    let linear = |input, output, salt| OwnedVisionLinearParameters {
        weight: compact_values(input * output, layer, salt, 64.0, 0.0),
        bias: compact_values(output, layer, salt + 4, 256.0, 0.0),
    };
    OwnedVisionEncoderLayerParameters {
        norm1: norm(11, 1.0),
        query: linear(HIDDEN, HIDDEN, 19),
        key: linear(HIDDEN, HIDDEN, 29),
        value: linear(HIDDEN, HIDDEN, 37),
        attention_output: linear(HIDDEN, HIDDEN, 43),
        norm2: norm(53, 1.0),
        mlp_fc1: linear(HIDDEN, INTERMEDIATE, 61),
        mlp_fc2: linear(INTERMEDIATE, HIDDEN, 71),
    }
}

fn compact_values(elements: usize, layer: u32, salt: u32, divisor: f32, shift: f32) -> Vec<f32> {
    (0..elements)
        .map(|index| {
            let residue = ((index as u32 * (salt + 11) + layer * 37 + salt * 13) % 31) as i32;
            (residue - 15) as f32 / divisor + shift
        })
        .collect()
}

fn compact_cpu_parameters(
    parameters: &OwnedVisionEncoderLayerParameters,
) -> CpuVisionLayerParameters<'_> {
    CpuVisionLayerParameters {
        norm1: CpuLayerNormParameters {
            weight: &parameters.norm1.weight,
            bias: &parameters.norm1.bias,
        },
        query: compact_cpu_linear(&parameters.query),
        key: compact_cpu_linear(&parameters.key),
        value: compact_cpu_linear(&parameters.value),
        attention_output: compact_cpu_linear(&parameters.attention_output),
        norm2: CpuLayerNormParameters {
            weight: &parameters.norm2.weight,
            bias: &parameters.norm2.bias,
        },
        mlp_fc1: compact_cpu_linear(&parameters.mlp_fc1),
        mlp_fc2: compact_cpu_linear(&parameters.mlp_fc2),
    }
}

fn compact_cpu_linear(parameters: &OwnedVisionLinearParameters) -> CpuLinearParameters<'_> {
    CpuLinearParameters {
        weight: &parameters.weight,
        bias: &parameters.bias,
    }
}

fn vision_layer_parameter_bytes(parameters: &OwnedVisionEncoderLayerParameters) -> Vec<u8> {
    let tensors: [&[f32]; 16] = [
        &parameters.norm1.weight,
        &parameters.norm1.bias,
        &parameters.query.weight,
        &parameters.query.bias,
        &parameters.key.weight,
        &parameters.key.bias,
        &parameters.value.weight,
        &parameters.value.bias,
        &parameters.attention_output.weight,
        &parameters.attention_output.bias,
        &parameters.norm2.weight,
        &parameters.norm2.bias,
        &parameters.mlp_fc1.weight,
        &parameters.mlp_fc1.bias,
        &parameters.mlp_fc2.weight,
        &parameters.mlp_fc2.bias,
    ];
    let mut bytes = Vec::new();
    for tensor in tensors {
        bytes.extend_from_slice(&f32_le_bytes(tensor));
    }
    bytes
}

fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn prepare_official_vision_layer(pack_path: &Path) -> Result<(), HarnessError> {
    if !matches!(env::var("PVLC_REQUIRE_MODEL").as_deref(), Ok("1" | "true")) {
        return Err(HarnessError::MissingGate("PVLC_REQUIRE_MODEL"));
    }
    let (invocation, expected) = load_official_vision_layer_fixture()?;
    let pack = build_vision_layer_self_test_pack(
        OFFICIAL_COMPILER_BUILD,
        VisionLayerSelfTestSource::official_layer_zero(invocation.borrowed(), &expected),
    )?;
    write_bytes_atomic(pack_path, &pack)
}

struct OfficialProjectorFixture {
    parameters: OwnedProjectorParameters,
    l3_input: Vec<f32>,
    l3_expected: BTreeMap<ProjectorStage, Vec<f32>>,
    l2_input: Vec<f32>,
    l2_expected: BTreeMap<ProjectorStage, Vec<f32>>,
}

fn prepare_official_projector(pack_path: &Path) -> Result<(), HarnessError> {
    if env::var("PVLC_REQUIRE_MODEL").as_deref() != Ok("1") {
        return Err(HarnessError::MissingGate("PVLC_REQUIRE_MODEL"));
    }
    let fixture = load_official_projector_fixture()?;
    let l3_grid = [[1, 22, 58]];
    let l2_grid = [[1, 30, 58]];
    let cases = [
        ProjectorSelfTestCaseSource::official_l3(&fixture.l3_input, &l3_grid, &fixture.l3_expected),
        ProjectorSelfTestCaseSource::official_l2(&fixture.l2_input, &l2_grid, &fixture.l2_expected),
    ];
    let pack = build_projector_self_test_pack(
        OFFICIAL_COMPILER_BUILD,
        ProjectorSelfTestSource::official(fixture.parameters.borrowed(), &cases),
    )?;
    write_bytes_atomic(pack_path, &pack)
}

fn load_official_projector_fixture() -> Result<OfficialProjectorFixture, HarnessError> {
    const HIDDEN: u64 = 1_152;
    const MERGED: u64 = HIDDEN * 4;
    const OUTPUT: u64 = 1_024;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model = SafetensorsCatalog::open(
        root.join("models/snapshots")
            .join(MODEL_REVISION)
            .join("model.safetensors"),
    )?;
    let parameters = OwnedProjectorParameters {
        pre_norm: OwnedVisionLayerNormParameters {
            weight: load_official_tensor(&model, "mlp_AR.pre_norm.weight", &[HIDDEN])?,
            bias: load_official_tensor(&model, "mlp_AR.pre_norm.bias", &[HIDDEN])?,
        },
        linear1: OwnedVisionLinearParameters {
            weight: load_official_tensor(&model, "mlp_AR.linear_1.weight", &[MERGED, MERGED])?,
            bias: load_official_tensor(&model, "mlp_AR.linear_1.bias", &[MERGED])?,
        },
        linear2: OwnedVisionLinearParameters {
            weight: load_official_tensor(&model, "mlp_AR.linear_2.weight", &[OUTPUT, MERGED])?,
            bias: load_official_tensor(&model, "mlp_AR.linear_2.bias", &[OUTPUT])?,
        },
    };

    let l3_root = root.join("artifacts/goldens/ocr.clean_latin.0001-l3");
    let l3_stage = SafetensorsCatalog::open(l3_root.join("stage-checkpoints.safetensors"))?;
    let l3_deep = SafetensorsCatalog::open(l3_root.join("deep-checkpoints.safetensors"))?;
    let l3_input = load_official_tensor(&l3_stage, "vision.final", &[1_276, HIDDEN])?;
    let l3_expected = BTreeMap::from([
        (
            ProjectorStage::PreNorm,
            load_official_tensor(&l3_deep, "projector.pre_norm", &[1_276, HIDDEN])?,
        ),
        (
            ProjectorStage::Merge,
            load_official_tensor(&l3_deep, "projector.merge", &[319, MERGED])?,
        ),
        (
            ProjectorStage::Linear1,
            load_official_tensor(&l3_deep, "projector.linear1", &[319, MERGED])?,
        ),
        (
            ProjectorStage::Activation,
            load_official_tensor(&l3_deep, "projector.gelu", &[319, MERGED])?,
        ),
        (
            ProjectorStage::Linear2,
            load_official_tensor(&l3_deep, "projector.linear2", &[319, OUTPUT])?,
        ),
    ]);

    let l2_root = root.join("artifacts/goldens/table.simple.0001-l2");
    let l2_stage = SafetensorsCatalog::open(l2_root.join("stage-checkpoints.safetensors"))?;
    let l2_input = load_official_tensor(&l2_stage, "vision.final", &[1_740, HIDDEN])?;
    let l2_expected = BTreeMap::from([(
        ProjectorStage::Linear2,
        load_official_tensor(&l2_stage, "projector.final", &[435, OUTPUT])?,
    )]);
    Ok(OfficialProjectorFixture {
        parameters,
        l3_input,
        l3_expected,
        l2_input,
        l2_expected,
    })
}

fn load_official_vision_layer_fixture() -> Result<
    (
        OwnedVisionEncoderLayerInvocation,
        BTreeMap<VisionEncoderLayerStage, Vec<f32>>,
    ),
    HarnessError,
> {
    const TOKENS: u64 = 1_276;
    const HIDDEN: u64 = 1_152;
    const INTERMEDIATE: u64 = 4_304;
    const PREFIX: &str = "visual.vision_model.encoder.layers.0";
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model = SafetensorsCatalog::open(
        root.join("models/snapshots")
            .join(MODEL_REVISION)
            .join("model.safetensors"),
    )?;
    let deep = SafetensorsCatalog::open(
        root.join("artifacts/goldens/ocr.clean_latin.0001-l3")
            .join("deep-checkpoints.safetensors"),
    )?;
    let linear = |weight: &str,
                  bias: &str,
                  shape: &[u64],
                  bias_shape: &[u64]|
     -> Result<OwnedVisionLinearParameters, HarnessError> {
        Ok(OwnedVisionLinearParameters {
            weight: load_official_tensor(&model, &format!("{PREFIX}.{weight}"), shape)?,
            bias: load_official_tensor(&model, &format!("{PREFIX}.{bias}"), bias_shape)?,
        })
    };
    let invocation = OwnedVisionEncoderLayerInvocation {
        tokens: TOKENS as u32,
        hidden_size: HIDDEN as u32,
        attention_heads: 16,
        head_dim: 72,
        intermediate_size: INTERMEDIATE as u32,
        layer_norm_epsilon: 1.0e-6,
        input: load_official_tensor(&deep, "vision.embeddings.output", &[1, TOKENS, HIDDEN])?,
        cu_seqlens: vec![0, TOKENS as u32],
        parameters: OwnedVisionEncoderLayerParameters {
            norm1: OwnedVisionLayerNormParameters {
                weight: load_official_tensor(
                    &model,
                    &format!("{PREFIX}.layer_norm1.weight"),
                    &[HIDDEN],
                )?,
                bias: load_official_tensor(
                    &model,
                    &format!("{PREFIX}.layer_norm1.bias"),
                    &[HIDDEN],
                )?,
            },
            query: linear(
                "self_attn.q_proj.weight",
                "self_attn.q_proj.bias",
                &[HIDDEN, HIDDEN],
                &[HIDDEN],
            )?,
            key: linear(
                "self_attn.k_proj.weight",
                "self_attn.k_proj.bias",
                &[HIDDEN, HIDDEN],
                &[HIDDEN],
            )?,
            value: linear(
                "self_attn.v_proj.weight",
                "self_attn.v_proj.bias",
                &[HIDDEN, HIDDEN],
                &[HIDDEN],
            )?,
            attention_output: linear(
                "self_attn.out_proj.weight",
                "self_attn.out_proj.bias",
                &[HIDDEN, HIDDEN],
                &[HIDDEN],
            )?,
            norm2: OwnedVisionLayerNormParameters {
                weight: load_official_tensor(
                    &model,
                    &format!("{PREFIX}.layer_norm2.weight"),
                    &[HIDDEN],
                )?,
                bias: load_official_tensor(
                    &model,
                    &format!("{PREFIX}.layer_norm2.bias"),
                    &[HIDDEN],
                )?,
            },
            mlp_fc1: linear(
                "mlp.fc1.weight",
                "mlp.fc1.bias",
                &[INTERMEDIATE, HIDDEN],
                &[INTERMEDIATE],
            )?,
            mlp_fc2: linear(
                "mlp.fc2.weight",
                "mlp.fc2.bias",
                &[HIDDEN, INTERMEDIATE],
                &[HIDDEN],
            )?,
        },
    };
    let names = [
        "vision.layer.00.norm1",
        "vision.layer.00.q",
        "vision.layer.00.k",
        "vision.layer.00.v",
        "vision.layer.00.attention.context",
        "vision.layer.00.attention.output",
        "vision.layer.00.attention.residual",
        "vision.layer.00.norm2",
        "vision.layer.00.mlp.fc1",
        "vision.layer.00.mlp.activation",
        "vision.layer.00.mlp.output",
        "vision.layer.00.output",
    ];
    let mut expected = BTreeMap::new();
    for (stage, name) in VisionEncoderLayerStage::ALL.into_iter().zip(names) {
        let width = if matches!(
            stage,
            VisionEncoderLayerStage::MlpFc1 | VisionEncoderLayerStage::MlpActivation
        ) {
            INTERMEDIATE
        } else {
            HIDDEN
        };
        expected.insert(
            stage,
            load_official_tensor(&deep, name, &[1, TOKENS, width])?,
        );
    }
    Ok((invocation, expected))
}

fn load_official_tensor(
    catalog: &SafetensorsCatalog,
    name: &str,
    expected_shape: &[u64],
) -> Result<Vec<f32>, HarnessError> {
    let tensor = catalog
        .tensor(name)
        .ok_or_else(|| HarnessError::MissingOfficialTensor(name.to_owned()))?;
    if tensor.shape != expected_shape {
        return Err(HarnessError::OfficialTensorShape {
            name: name.to_owned(),
            actual: tensor.shape.clone(),
            expected: expected_shape.to_vec(),
        });
    }
    Ok(catalog.load_tensor_f32(name)?)
}

fn projector_values(length: usize, seed: u32) -> Vec<f32> {
    (0..length)
        .map(|index| {
            let residue = (index as u32 * 17 + seed * 13) % 97;
            (residue as i32 - 48) as f32 / 64.0
        })
        .collect()
}

fn projector_invocation(poisoned_image: Option<usize>) -> OwnedProjectorInvocation {
    let mut input = projector_values(16 * PROJECTOR_HIDDEN, 1);
    if let Some(image) = poisoned_image {
        let start = image * 8 * PROJECTOR_HIDDEN;
        let channel_offsets = [1.5_f32, -2.25, 3.0];
        for (local, value) in input[start..start + 8 * PROJECTOR_HIDDEN]
            .iter_mut()
            .enumerate()
        {
            *value = -*value * 4.0 + channel_offsets[local % PROJECTOR_HIDDEN];
        }
    }
    OwnedProjectorInvocation {
        hidden_size: PROJECTOR_HIDDEN as u32,
        output_size: PROJECTOR_OUTPUT as u32,
        layer_norm_epsilon: PROJECTOR_EPSILON,
        input,
        image_grid_thw: PROJECTOR_GRIDS.to_vec(),
        parameters: OwnedProjectorParameters {
            pre_norm: OwnedVisionLayerNormParameters {
                weight: projector_values(PROJECTOR_HIDDEN, 11)
                    .into_iter()
                    .map(|value| value + 1.0)
                    .collect(),
                bias: projector_values(PROJECTOR_HIDDEN, 12),
            },
            linear1: OwnedVisionLinearParameters {
                weight: projector_values(PROJECTOR_MERGED * PROJECTOR_MERGED, 13),
                bias: projector_values(PROJECTOR_MERGED, 14),
            },
            linear2: OwnedVisionLinearParameters {
                weight: projector_values(PROJECTOR_OUTPUT * PROJECTOR_MERGED, 15),
                bias: projector_values(PROJECTOR_OUTPUT, 16),
            },
        },
    }
}

fn projector_cpu_trace(
    invocation: &OwnedProjectorInvocation,
) -> Result<CpuProjectorTrace, HarnessError> {
    let parameters = &invocation.parameters;
    let grids = PROJECTOR_GRIDS.map(|grid| grid.map(|dimension| dimension as usize));
    cpu_projector_f32(
        &invocation.input,
        PROJECTOR_HIDDEN,
        &grids,
        CpuProjectorParameters {
            pre_norm: CpuLayerNormParameters {
                weight: &parameters.pre_norm.weight,
                bias: &parameters.pre_norm.bias,
            },
            linear1: CpuLinearParameters {
                weight: &parameters.linear1.weight,
                bias: &parameters.linear1.bias,
            },
            linear2: CpuLinearParameters {
                weight: &parameters.linear2.weight,
                bias: &parameters.linear2.bias,
            },
        },
        PROJECTOR_EPSILON,
    )
    .map_err(HarnessError::ProjectorCpu)
}

fn projector_checkpoints(trace: CpuProjectorTrace) -> BTreeMap<ProjectorStage, Vec<f32>> {
    BTreeMap::from([
        (ProjectorStage::PreNorm, trace.pre_norm),
        (ProjectorStage::Merge, trace.merged),
        (ProjectorStage::Linear1, trace.linear1),
        (ProjectorStage::Activation, trace.activation),
        (ProjectorStage::Linear2, trace.output),
    ])
}

fn compact_projector_cases() -> Result<Vec<ProjectorCase>, HarnessError> {
    [None, Some(0), Some(1)]
        .into_iter()
        .enumerate()
        .map(|(index, poisoned_image)| {
            let invocation = projector_invocation(poisoned_image);
            let expected = projector_checkpoints(projector_cpu_trace(&invocation)?);
            Ok(ProjectorCase {
                id: if index == 0 {
                    "projector/baseline".to_owned()
                } else {
                    format!("projector/poison-image-{}", index - 1)
                },
                poisoned_image,
                invocation,
                expected,
                policy: PROJECTOR_POLICY,
            })
        })
        .collect()
}

fn prepare_projector(corpus_path: &Path, native_baseline_path: &Path) -> Result<(), HarnessError> {
    require_hardware_gates()?;
    let cases = compact_projector_cases()?;
    let source = ProjectorCorpusSource {
        schema_version: 1,
        oracle: PROJECTOR_ORACLE,
        cases: &cases,
    };
    let canonical = serde_json::to_vec(&source).map_err(|source| HarnessError::Serialize {
        path: corpus_path.to_path_buf(),
        source,
    })?;
    let corpus_blake3 = blake3::hash(&canonical).to_hex().to_string();
    let runtime = NativeRuntime::new(NativeOptions::default())?;
    let baseline = build_projector_native_baseline(&runtime, &cases, corpus_blake3.clone())?;
    let artifact = ProjectorCorpusArtifact {
        schema_version: 1,
        oracle: PROJECTOR_ORACLE,
        corpus_blake3: &corpus_blake3,
        cases: &cases,
    };
    write_json_atomic(corpus_path, &artifact)?;
    write_json_atomic(native_baseline_path, &baseline)?;
    Ok(())
}

fn build_projector_native_baseline(
    runtime: &NativeRuntime,
    cases: &[ProjectorCase],
    corpus_blake3: String,
) -> Result<ProjectorNativeBaseline, HarnessError> {
    require_native_capabilities(runtime)?;
    let before = runtime.counters();
    let mut native_cases = Vec::with_capacity(cases.len());
    for case in cases {
        let execution =
            runtime.run_projector(&case.invocation.borrowed(), ProjectorReadback::AllStages)?;
        for stage in ProjectorStage::ALL {
            let expected = &case.expected[&stage];
            let actual =
                execution
                    .checkpoints
                    .get(&stage)
                    .ok_or_else(|| HarnessError::InvalidEvidence {
                        case_id: case.id.clone(),
                        message: format!("native projector omitted {stage:?}"),
                    })?;
            let report = compare_f32(
                expected,
                actual,
                &[expected.len()],
                ComparisonAxes::default(),
            )
            .map_err(|error| HarnessError::CpuMismatch {
                case_id: case.id.clone(),
                message: format!("{stage:?}: {error}"),
            })?;
            let verdict = report
                .assess(&case.policy.native_comparison_policy())
                .map_err(|error| HarnessError::CpuMismatch {
                    case_id: case.id.clone(),
                    message: format!("{stage:?}: {error}"),
                })?;
            if !verdict.passed() {
                return Err(HarnessError::CpuMismatch {
                    case_id: case.id.clone(),
                    message: format!(
                        "{stage:?}: {report:?}; violations: {:?}",
                        verdict.violations()
                    ),
                });
            }
        }
        native_cases.push(ProjectorNativeCase {
            id: case.id.clone(),
            checkpoints: execution.checkpoints,
            diagnostics: projector_diagnostics(&case.id, execution.diagnostics)?,
        });
    }
    let after = runtime.counters();
    let submission_count = after.submissions.saturating_sub(before.submissions);
    if submission_count != native_cases.len() as u64 {
        return Err(HarnessError::InvalidEvidence {
            case_id: "<projector-aggregate>".to_owned(),
            message: format!(
                "expected exactly {} projector submissions, observed {submission_count}",
                native_cases.len()
            ),
        });
    }
    Ok(ProjectorNativeBaseline {
        schema_version: 1,
        corpus_blake3,
        capabilities: native_capabilities_report(runtime),
        submission_count,
        cases: native_cases,
    })
}

fn projector_diagnostics(
    case_id: &str,
    diagnostics: ProjectorDiagnostics,
) -> Result<ProjectorNativeDiagnosticsReport, HarnessError> {
    if diagnostics.checked_error_scopes
        != [
            ErrorScopeKind::Validation,
            ErrorScopeKind::OutOfMemory,
            ErrorScopeKind::Internal,
        ]
    {
        return invalid_evidence(case_id, "projector error scopes are missing or reordered");
    }
    if !diagnostics.captured_errors.is_empty()
        || diagnostics.queue_wall_time_ns == 0
        || diagnostics.dispatch_stages != ProjectorStage::ALL
        || diagnostics.submission_count != 1
        || diagnostics.command_buffer_count != 1
        || diagnostics.compute_pass_count != 1
        || diagnostics.dispatch_count != 5
        || diagnostics.buffer_allocation_count != 15
        || diagnostics.readback_buffer_count != 1
        || diagnostics.readback_map_count != 1
        || diagnostics.readback_bytes != 848
        || diagnostics.resident_intermediate_bytes != 848
        || diagnostics.resident_weight_bytes != 908
    {
        return invalid_evidence(case_id, "projector residency diagnostics are invalid");
    }
    let timestamp = diagnostics
        .timestamp
        .ok_or_else(|| HarnessError::InvalidEvidence {
            case_id: case_id.to_owned(),
            message: "projector timestamp query is missing".to_owned(),
        })?;
    validate_timestamp(case_id, timestamp)?;
    if diagnostics.timestamp_fresh != Some(true) {
        return invalid_evidence(
            case_id,
            &format!(
                "projector timestamp query freshness is {:?}",
                diagnostics.timestamp_fresh
            ),
        );
    }
    let shader_blake3 = diagnostics
        .shader_blake3
        .into_iter()
        .map(|(kernel, hash)| (kernel, blake3::Hash::from_bytes(hash).to_hex().to_string()))
        .collect::<BTreeMap<_, _>>();
    if shader_blake3.len() != 4 {
        return invalid_evidence(case_id, "projector shader hash set is incomplete");
    }
    Ok(ProjectorNativeDiagnosticsReport {
        checked_error_scopes: ["validation", "out_of_memory", "internal"],
        captured_errors: diagnostics.captured_errors,
        queue_wall_time_ns: diagnostics.queue_wall_time_ns,
        timestamp: ProjectorNativeTimestampReport {
            begin_ticks: timestamp.begin_ticks,
            end_ticks: timestamp.end_ticks,
            period_ns: timestamp.period_ns,
            duration_ns: timestamp.duration_ns,
        },
        timestamp_fresh: true,
        shader_blake3,
        dispatch_stages: diagnostics.dispatch_stages,
        submission_count: diagnostics.submission_count,
        command_buffer_count: diagnostics.command_buffer_count,
        compute_pass_count: 1,
        dispatch_count: 5,
        buffer_allocation_count: diagnostics.buffer_allocation_count,
        readback_buffer_count: diagnostics.readback_buffer_count,
        readback_map_count: diagnostics.readback_map_count,
        readback_bytes: diagnostics.readback_bytes,
        resident_intermediate_bytes: diagnostics.resident_intermediate_bytes,
        resident_weight_bytes: diagnostics.resident_weight_bytes,
    })
}

fn prepare_vision_layer(
    corpus_path: &Path,
    native_baseline_path: &Path,
) -> Result<(), HarnessError> {
    require_hardware_gates()?;
    let corpus = m3_vision_layer_corpus()?;
    let canonical = serde_json::to_vec(&corpus).map_err(|source| HarnessError::Serialize {
        path: corpus_path.to_path_buf(),
        source,
    })?;
    let corpus_blake3 = blake3::hash(&canonical).to_hex().to_string();
    let runtime = NativeRuntime::new(NativeOptions::default())?;
    let baseline = build_vision_layer_native_baseline(&runtime, &corpus, corpus_blake3.clone())?;
    let artifact = VisionLayerCorpusArtifact {
        schema_version: corpus.schema_version,
        oracle: &corpus.oracle,
        fixture_algorithm: &corpus.fixture_algorithm,
        corpus_blake3: &corpus_blake3,
        cases: &corpus.cases,
    };
    write_json_atomic(corpus_path, &artifact)?;
    write_json_atomic(native_baseline_path, &baseline)?;
    Ok(())
}

fn prepare_vision_attention(
    corpus_path: &Path,
    native_baseline_path: &Path,
) -> Result<(), HarnessError> {
    require_hardware_gates()?;
    let corpus = m3_vision_attention_corpus()?;
    let canonical = serde_json::to_vec(&corpus).map_err(|source| HarnessError::Serialize {
        path: corpus_path.to_path_buf(),
        source,
    })?;
    let corpus_blake3 = blake3::hash(&canonical).to_hex().to_string();
    let runtime = NativeRuntime::new(NativeOptions::default())?;
    let baseline = build_vision_native_baseline(&runtime, &corpus, corpus_blake3.clone())?;
    let corpus_artifact = VisionCorpusArtifact {
        schema_version: corpus.schema_version,
        oracle: &corpus.oracle,
        fixture_algorithm: &corpus.fixture_algorithm,
        corpus_blake3: &corpus_blake3,
        cases: &corpus.cases,
    };
    write_json_atomic(corpus_path, &corpus_artifact)?;
    write_json_atomic(native_baseline_path, &baseline)?;
    Ok(())
}

fn prepare(corpus_path: &Path, native_baseline_path: &Path) -> Result<(), HarnessError> {
    require_hardware_gates()?;
    let corpus = m2_primitive_corpus()?;
    let canonical = serde_json::to_vec(&corpus).map_err(|source| HarnessError::Serialize {
        path: corpus_path.to_path_buf(),
        source,
    })?;
    let corpus_blake3 = blake3::hash(&canonical).to_hex().to_string();
    let runtime = NativeRuntime::new(NativeOptions::default())?;
    let baseline = build_native_baseline(&runtime, &corpus, corpus_blake3.clone())?;
    let corpus_artifact = CorpusArtifact {
        schema_version: corpus.schema_version,
        oracle: &corpus.oracle,
        corpus_blake3: &corpus_blake3,
        cases: &corpus.cases,
    };
    write_json_atomic(corpus_path, &corpus_artifact)?;
    write_json_atomic(native_baseline_path, &baseline)?;
    Ok(())
}

fn require_hardware_gates() -> Result<(), HarnessError> {
    for gate in REQUIRED_GATES {
        if !matches!(env::var(gate).as_deref(), Ok("1" | "true")) {
            return Err(HarnessError::MissingGate(gate));
        }
    }
    Ok(())
}

fn build_native_baseline(
    runtime: &NativeRuntime,
    corpus: &M2PrimitiveCorpus,
    corpus_blake3: String,
) -> Result<NativeBaseline, HarnessError> {
    require_native_capabilities(runtime)?;
    let before = runtime.counters();
    let mut cases = Vec::with_capacity(corpus.cases.len());
    for case in &corpus.cases {
        cases.push(execute_native_case(
            runtime,
            &case.id,
            case.kernel,
            &case.invocation,
            &case.expected,
            &case.shape,
            case.policy.comparison_policy(),
        )?);
    }
    finish_native_baseline(runtime, before.submissions, cases, corpus_blake3)
}

fn build_vision_native_baseline(
    runtime: &NativeRuntime,
    corpus: &M3VisionAttentionCorpus,
    corpus_blake3: String,
) -> Result<NativeBaseline, HarnessError> {
    require_native_capabilities(runtime)?;
    let before = runtime.counters();
    let mut cases = Vec::with_capacity(corpus.cases.len());
    for case in &corpus.cases {
        let invocation = case.invocation()?;
        cases.push(execute_native_case(
            runtime,
            &case.id,
            KernelId::VisionAttentionF32,
            &invocation,
            &case.expected,
            &case.shape,
            case.policy.native_comparison_policy(),
        )?);
    }
    finish_native_baseline(runtime, before.submissions, cases, corpus_blake3)
}

fn build_vision_layer_native_baseline(
    runtime: &NativeRuntime,
    corpus: &M3VisionLayerCorpus,
    corpus_blake3: String,
) -> Result<VisionLayerNativeBaseline, HarnessError> {
    require_native_capabilities(runtime)?;
    let before = runtime.counters();
    let mut cases = Vec::with_capacity(corpus.cases.len());
    for case in &corpus.cases {
        let invocation = case.invocation()?;
        let execution = runtime.run_vision_encoder_layer_identity_rope(
            &invocation.borrowed(),
            VisionLayerReadback::AllStages,
        )?;
        for stage in VisionEncoderLayerStage::ALL {
            let actual =
                execution
                    .checkpoints
                    .get(&stage)
                    .ok_or_else(|| HarnessError::InvalidEvidence {
                        case_id: case.id.clone(),
                        message: format!("native vision layer omitted {stage:?}"),
                    })?;
            let width = vision_layer_stage_width(case, stage);
            let report = compare_f32(
                case.expected.stage(stage),
                actual,
                &[case.tokens as usize, width],
                ComparisonAxes::default(),
            )
            .map_err(|error| HarnessError::CpuMismatch {
                case_id: case.id.clone(),
                message: format!("{stage:?}: {error}"),
            })?;
            let verdict = report
                .assess(&case.policy.native_comparison_policy())
                .map_err(|error| HarnessError::CpuMismatch {
                    case_id: case.id.clone(),
                    message: format!("{stage:?}: {error}"),
                })?;
            if !verdict.passed() {
                return Err(HarnessError::CpuMismatch {
                    case_id: case.id.clone(),
                    message: format!(
                        "{stage:?}: {report:?}; violations: {:?}",
                        verdict.violations()
                    ),
                });
            }
        }
        cases.push(VisionLayerNativeCase {
            id: case.id.clone(),
            checkpoints: execution.checkpoints,
            diagnostics: vision_layer_diagnostics(&case.id, execution.diagnostics)?,
        });
    }
    let after = runtime.counters();
    let submission_count = after.submissions.saturating_sub(before.submissions);
    if submission_count != cases.len() as u64 {
        return Err(HarnessError::InvalidEvidence {
            case_id: "<vision-layer-aggregate>".to_owned(),
            message: format!(
                "expected exactly {} native submissions, observed {submission_count}",
                cases.len()
            ),
        });
    }
    Ok(VisionLayerNativeBaseline {
        schema_version: 1,
        corpus_blake3,
        capabilities: native_capabilities_report(runtime),
        submission_count,
        cases,
    })
}

fn vision_layer_stage_width(case: &M3VisionLayerCase, stage: VisionEncoderLayerStage) -> usize {
    if matches!(
        stage,
        VisionEncoderLayerStage::MlpFc1 | VisionEncoderLayerStage::MlpActivation
    ) {
        case.intermediate_size as usize
    } else {
        case.hidden_size as usize
    }
}

fn vision_layer_diagnostics(
    case_id: &str,
    diagnostics: VisionLayerDiagnostics,
) -> Result<VisionLayerNativeDiagnosticsReport, HarnessError> {
    if diagnostics.checked_error_scopes
        != [
            ErrorScopeKind::Validation,
            ErrorScopeKind::OutOfMemory,
            ErrorScopeKind::Internal,
        ]
    {
        return invalid_evidence(
            case_id,
            "vision-layer error scopes are missing or reordered",
        );
    }
    if !diagnostics.captured_errors.is_empty() || diagnostics.queue_wall_time_ns == 0 {
        return invalid_evidence(
            case_id,
            "vision-layer captured an error or has zero queue timing",
        );
    }
    if diagnostics.dispatch_stages != VisionEncoderLayerStage::ALL
        || diagnostics.submission_count != 1
        || diagnostics.command_buffer_count != 1
        || diagnostics.buffer_allocation_count == 0
        || diagnostics.buffer_allocation_count > 32
        || diagnostics.readback_buffer_count != 1
    {
        return invalid_evidence(case_id, "vision-layer residency diagnostics are invalid");
    }
    let timestamp = diagnostics
        .timestamp
        .ok_or_else(|| HarnessError::InvalidEvidence {
            case_id: case_id.to_owned(),
            message: "vision-layer timestamp query is missing".to_owned(),
        })?;
    validate_timestamp(case_id, timestamp)?;
    let shader_blake3 = diagnostics
        .shader_blake3
        .into_iter()
        .map(|(kernel, hash)| (kernel, blake3::Hash::from_bytes(hash).to_hex().to_string()))
        .collect::<BTreeMap<_, _>>();
    if shader_blake3.len() != 5 {
        return invalid_evidence(case_id, "vision-layer shader hash set is incomplete");
    }
    Ok(VisionLayerNativeDiagnosticsReport {
        checked_error_scopes: ["validation", "out_of_memory", "internal"],
        captured_errors: diagnostics.captured_errors,
        queue_wall_time_ns: diagnostics.queue_wall_time_ns,
        shader_blake3,
        dispatch_stages: diagnostics.dispatch_stages,
        rope_specialization: diagnostics.rope_specialization,
        submission_count: diagnostics.submission_count,
        command_buffer_count: diagnostics.command_buffer_count,
        compute_pass_count: 1,
        dispatch_count: 12,
        buffer_allocation_count: diagnostics.buffer_allocation_count,
        readback_buffer_count: diagnostics.readback_buffer_count,
        readback_bytes: diagnostics.readback_bytes,
        timestamp: NativeTimestampReport {
            begin_ticks: timestamp.begin_ticks,
            end_ticks: timestamp.end_ticks,
            duration_ns: timestamp.duration_ns,
        },
    })
}

fn require_native_capabilities(runtime: &NativeRuntime) -> Result<(), HarnessError> {
    let capabilities = runtime.capabilities();
    if capabilities.backend != BackendKind::Metal || !capabilities.adapter_name.contains("M4 Pro") {
        return Err(HarnessError::WrongAdapter {
            backend: capabilities.backend,
            adapter: capabilities.adapter_name.clone(),
        });
    }
    if !capabilities.timestamp_query {
        return Err(HarnessError::TimestampUnavailable);
    }
    Ok(())
}

fn execute_native_case(
    runtime: &NativeRuntime,
    case_id: &str,
    kernel: KernelId,
    invocation: &KernelInvocation,
    expected: &[f32],
    shape: &[usize],
    policy: ComparisonPolicy,
) -> Result<NativeCase, HarnessError> {
    let execution = runtime.run(invocation)?;
    let report = compare_f32(
        expected,
        &execution.values,
        shape,
        ComparisonAxes::default(),
    )
    .map_err(|error| HarnessError::CpuMismatch {
        case_id: case_id.to_owned(),
        message: error.to_string(),
    })?;
    let verdict = report
        .assess(&policy)
        .map_err(|error| HarnessError::CpuMismatch {
            case_id: case_id.to_owned(),
            message: error.to_string(),
        })?;
    if !verdict.passed() {
        return Err(HarnessError::CpuMismatch {
            case_id: case_id.to_owned(),
            message: format!("{report:?}; violations: {:?}", verdict.violations()),
        });
    }
    Ok(NativeCase {
        id: case_id.to_owned(),
        kernel,
        values: execution.values,
        diagnostics: diagnostics(case_id, kernel, execution.diagnostics)?,
    })
}

fn finish_native_baseline(
    runtime: &NativeRuntime,
    submissions_before: u64,
    cases: Vec<NativeCase>,
    corpus_blake3: String,
) -> Result<NativeBaseline, HarnessError> {
    let after = runtime.counters();
    let submission_count = after.submissions.saturating_sub(submissions_before);
    if submission_count < cases.len() as u64 {
        return Err(HarnessError::InvalidEvidence {
            case_id: "<aggregate>".to_owned(),
            message: format!(
                "only {submission_count} submissions were observed for {} cases",
                cases.len()
            ),
        });
    }

    Ok(NativeBaseline {
        schema_version: 1,
        corpus_blake3,
        capabilities: native_capabilities_report(runtime),
        submission_count,
        cases,
    })
}

fn native_capabilities_report(runtime: &NativeRuntime) -> NativeCapabilitiesReport {
    let capabilities = runtime.capabilities();
    NativeCapabilitiesReport {
        adapter_name: capabilities.adapter_name.clone(),
        backend: "metal",
        timestamp_query: capabilities.timestamp_query,
        max_storage_buffer_binding_size: capabilities.max_storage_buffer_binding_size,
        max_compute_workgroups_per_dimension: capabilities.max_compute_workgroups_per_dimension,
        max_compute_invocations_per_workgroup: capabilities.max_compute_invocations_per_workgroup,
        max_compute_workgroup_size_x: capabilities.max_compute_workgroup_size_x,
        max_compute_workgroup_size_y: capabilities.max_compute_workgroup_size_y,
        max_compute_workgroup_size_z: capabilities.max_compute_workgroup_size_z,
        max_compute_workgroup_storage_size: capabilities.max_compute_workgroup_storage_size,
        max_storage_buffers_per_shader_stage: capabilities.max_storage_buffers_per_shader_stage,
        max_buffer_size: capabilities.max_buffer_size,
    }
}

fn diagnostics(
    case_id: &str,
    kernel: KernelId,
    diagnostics: KernelDiagnostics,
) -> Result<NativeDiagnosticsReport, HarnessError> {
    if diagnostics.kernel != kernel {
        return invalid_evidence(case_id, "diagnostics kernel mismatch");
    }
    if diagnostics.checked_error_scopes
        != [
            ErrorScopeKind::Validation,
            ErrorScopeKind::OutOfMemory,
            ErrorScopeKind::Internal,
        ]
    {
        return invalid_evidence(case_id, "error scopes are missing or out of order");
    }
    if !diagnostics.captured_errors.is_empty() || diagnostics.queue_wall_time_ns == 0 {
        return invalid_evidence(case_id, "captured error or zero queue timing");
    }
    let timestamp = diagnostics
        .timestamp
        .ok_or_else(|| HarnessError::InvalidEvidence {
            case_id: case_id.to_owned(),
            message: "timestamp query result is missing".to_owned(),
        })?;
    validate_timestamp(case_id, timestamp)?;
    Ok(NativeDiagnosticsReport {
        checked_error_scopes: ["validation", "out_of_memory", "internal"],
        captured_errors: diagnostics.captured_errors,
        queue_wall_time_ns: diagnostics.queue_wall_time_ns,
        shader_blake3: blake3::Hash::from_bytes(diagnostics.shader_blake3)
            .to_hex()
            .to_string(),
        timestamp: NativeTimestampReport {
            begin_ticks: timestamp.begin_ticks,
            end_ticks: timestamp.end_ticks,
            duration_ns: timestamp.duration_ns,
        },
    })
}

fn validate_timestamp(case_id: &str, timestamp: GpuTimestamp) -> Result<(), HarnessError> {
    if timestamp.begin_ticks == 0
        || timestamp.end_ticks <= timestamp.begin_ticks
        || !timestamp.duration_ns.is_finite()
        || timestamp.duration_ns <= 0.0
    {
        return invalid_evidence(case_id, "timestamp query is zero, stale, or non-finite");
    }
    Ok(())
}

fn invalid_evidence<T>(case_id: &str, message: &str) -> Result<T, HarnessError> {
    Err(HarnessError::InvalidEvidence {
        case_id: case_id.to_owned(),
        message: message.to_owned(),
    })
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<(), HarnessError> {
    let parent = path
        .parent()
        .ok_or_else(|| HarnessError::MissingParent(path.to_path_buf()))?;
    fs::create_dir_all(parent).map_err(|source| HarnessError::CreateDirectory {
        path: parent.to_path_buf(),
        source,
    })?;
    let mut temporary =
        NamedTempFile::new_in(parent).map_err(|source| HarnessError::CreateTemporary {
            path: path.to_path_buf(),
            source,
        })?;
    {
        let mut writer = BufWriter::new(temporary.as_file_mut());
        serde_json::to_writer_pretty(&mut writer, value).map_err(|source| {
            HarnessError::Serialize {
                path: path.to_path_buf(),
                source,
            }
        })?;
        writer
            .write_all(b"\n")
            .map_err(|source| HarnessError::Flush {
                path: path.to_path_buf(),
                source,
            })?;
        writer.flush().map_err(|source| HarnessError::Flush {
            path: path.to_path_buf(),
            source,
        })?;
    }
    sync_file(temporary.as_file(), path)?;
    temporary
        .persist(path)
        .map_err(|source| HarnessError::Persist {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(())
}

fn write_bytes_atomic(path: &Path, bytes: &[u8]) -> Result<(), HarnessError> {
    let parent = path
        .parent()
        .ok_or_else(|| HarnessError::MissingParent(path.to_path_buf()))?;
    fs::create_dir_all(parent).map_err(|source| HarnessError::CreateDirectory {
        path: parent.to_path_buf(),
        source,
    })?;
    let mut temporary =
        NamedTempFile::new_in(parent).map_err(|source| HarnessError::CreateTemporary {
            path: path.to_path_buf(),
            source,
        })?;
    temporary
        .write_all(bytes)
        .and_then(|()| temporary.flush())
        .map_err(|source| HarnessError::Flush {
            path: path.to_path_buf(),
            source,
        })?;
    sync_file(temporary.as_file(), path)?;
    temporary
        .persist(path)
        .map_err(|source| HarnessError::Persist {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(())
}

fn sync_file(file: &File, path: &Path) -> Result<(), HarnessError> {
    file.sync_all().map_err(|source| HarnessError::Flush {
        path: path.to_path_buf(),
        source,
    })
}
