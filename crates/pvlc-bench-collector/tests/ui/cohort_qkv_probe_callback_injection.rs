use pvlc_bench_collector::{
    NativeBenchmarkLeafPlanV1, NativeBenchmarkVisionStackValidatorV1,
    run_native_public_qkv_benchmark_cohort_v1,
};
use pvlc_passes::VisionQkvStackSelection;
use pvlc_runtime_core::{VisionEncoderStackInvocation, VisionStackActivationStrategy};
use pvlc_runtime_native::NativeRuntime;

fn inject_probe_callback(
    runtime: &NativeRuntime,
    plan: NativeBenchmarkLeafPlanV1,
    invocation: &VisionEncoderStackInvocation<'_>,
    selection: &VisionQkvStackSelection,
    validator: &mut NativeBenchmarkVisionStackValidatorV1,
) {
    let _ = run_native_public_qkv_benchmark_cohort_v1(
        runtime,
        plan,
        invocation,
        &[0],
        VisionStackActivationStrategy::StaticArenaAlias,
        selection,
        || panic!("hostile QKV probe callback"),
        validator,
    );
}

fn main() {}
