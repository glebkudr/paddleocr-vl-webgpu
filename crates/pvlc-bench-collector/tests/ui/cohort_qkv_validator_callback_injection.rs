use pvlc_bench_collector::{
    NativeBenchmarkEnvironmentProbeV1, NativeBenchmarkLeafPlanV1,
    run_native_public_qkv_benchmark_cohort_v1,
};
use pvlc_passes::VisionQkvStackSelection;
use pvlc_runtime_core::{VisionEncoderStackInvocation, VisionStackActivationStrategy};
use pvlc_runtime_native::NativeRuntime;

fn inject_validator_callback(
    runtime: &NativeRuntime,
    plan: NativeBenchmarkLeafPlanV1,
    invocation: &VisionEncoderStackInvocation<'_>,
    selection: &VisionQkvStackSelection,
    probe: &mut NativeBenchmarkEnvironmentProbeV1,
) {
    let _ = run_native_public_qkv_benchmark_cohort_v1(
        runtime,
        plan,
        invocation,
        &[0],
        VisionStackActivationStrategy::StaticArenaAlias,
        selection,
        probe,
        |_| panic!("hostile QKV validator callback"),
    );
}

fn main() {}
