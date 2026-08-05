use pvlc_bench_collector::NativeBenchmarkCohortFailureV1;

fn escape(failure: &mut NativeBenchmarkCohortFailureV1) {
    let _ = &mut failure.phase;
}

fn main() {}
