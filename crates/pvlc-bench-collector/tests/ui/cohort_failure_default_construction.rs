use pvlc_bench_collector::NativeBenchmarkCohortFailureV1;

fn require_default<T: Default>() {}

fn main() {
    require_default::<NativeBenchmarkCohortFailureV1>();
}
