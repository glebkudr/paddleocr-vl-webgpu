use pvlc_bench_collector::NativeBenchmarkEnvironmentProbeV1;

fn require_default<T: Default>() {}

fn main() {
    require_default::<NativeBenchmarkEnvironmentProbeV1>();
}
