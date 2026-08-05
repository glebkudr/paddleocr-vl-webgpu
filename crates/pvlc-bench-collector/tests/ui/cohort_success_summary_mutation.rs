use pvlc_bench_collector::NativeBenchmarkCohortSuccessV1;

fn escape(success: &mut NativeBenchmarkCohortSuccessV1) {
    success.assembled().evidence().summary().count = 0;
}

fn main() {}
