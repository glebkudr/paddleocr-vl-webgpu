use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use pvlc_cli::{
    CompileOfficialVisionStackOptions, CompileTinyOptions, OfficialVisionStackProfile,
    compile_official_vision_stack_shards, compile_tiny_to_path,
};
use pvlc_pack::PrecisionProfile;
use pvlc_safetensors::convert_bf16_checkpoint_to_f16;

#[derive(Debug, Parser)]
#[command(name = "pvlc", about = "Paddle Vision-Language Compiler")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum VisionStackProfileArg {
    OcrCleanLatinL3,
    TableSimpleL2,
}

impl From<VisionStackProfileArg> for OfficialVisionStackProfile {
    fn from(profile: VisionStackProfileArg) -> Self {
        match profile {
            VisionStackProfileArg::OcrCleanLatinL3 => Self::OcrCleanLatinL3,
            VisionStackProfileArg::TableSimpleL2 => Self::TableSimpleL2,
        }
    }
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Convert one BF16 safetensors checkpoint to deterministic IEEE FP16.
    ConvertCheckpointFp16 {
        #[arg(long)]
        source: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Verify the pinned source and emit a deterministic M1 tiny pack.
    CompileTiny {
        #[arg(long)]
        lock: PathBuf,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        compiler_build: String,
        #[arg(long, default_value_t = 64)]
        context_limit: u32,
    },
    /// Compile the pinned official vision encoder into bounded browser shards.
    CompileOfficialVisionStackShards {
        #[arg(long, value_enum)]
        profile: VisionStackProfileArg,
        #[arg(long)]
        lock: PathBuf,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        golden_dir: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long)]
        compiler_build: String,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result: Result<(), String> = match cli.command {
        Command::ConvertCheckpointFp16 { source, output } => {
            convert_bf16_checkpoint_to_f16(source, output)
                .map_err(|error| error.to_string())
                .and_then(|report| {
                    serde_json::to_string(&report)
                        .map(|json| println!("{json}"))
                        .map_err(|error| format!("cannot encode conversion report: {error}"))
                })
        }
        Command::CompileTiny {
            lock,
            model_dir,
            output,
            compiler_build,
            context_limit,
        } => compile_tiny_to_path(
            lock,
            model_dir,
            output,
            &CompileTinyOptions {
                compiler_build,
                precision_profile: PrecisionProfile::Fidelity,
                resolution_buckets: vec![[28, 28]],
                context_limit,
            },
        )
        .map_err(|error| error.to_string()),
        Command::CompileOfficialVisionStackShards {
            profile,
            lock,
            model_dir,
            golden_dir,
            output_dir,
            compiler_build,
        } => compile_official_vision_stack_shards(
            lock,
            model_dir,
            golden_dir,
            output_dir,
            &CompileOfficialVisionStackOptions {
                compiler_build,
                profile: profile.into(),
            },
        )
        .map(|_| ())
        .map_err(|error| error.to_string()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
