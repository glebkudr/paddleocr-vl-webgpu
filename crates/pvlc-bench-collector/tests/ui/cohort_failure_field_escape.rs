use pvlc_bench_collector::NativeBenchmarkCohortFailureV1;

fn escape(failure: &mut NativeBenchmarkCohortFailureV1) {
    failure.run_id.clear();
}

fn main() {}
