use pvlc_bench_collector::NativeBenchmarkCohortSuccessV1;

fn escape(success: &mut NativeBenchmarkCohortSuccessV1) {
    success
        .assembled()
        .assembly_blake3()
        .make_ascii_uppercase();
}

fn main() {}
