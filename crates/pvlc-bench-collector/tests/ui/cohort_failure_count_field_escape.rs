use pvlc_bench_collector::NativeBenchmarkCohortFailureV1;

fn escape(failure: &mut NativeBenchmarkCohortFailureV1) {
    failure.expected_attempt_count = 0;
}

fn main() {}
