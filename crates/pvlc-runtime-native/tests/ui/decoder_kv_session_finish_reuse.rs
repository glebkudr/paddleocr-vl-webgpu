use pvlc_runtime_native::DecoderKvSessionStep;
use pvlc_runtime_native::NativeDecoderKvSession;

fn reuse_after_finish(
    mut session: NativeDecoderKvSession<'_>,
    step: &DecoderKvSessionStep<'_>,
) {
    let _ = session.finish();
    let _ = session.cache_tokens();
    let _ = session.step(step);
    let _ = session.finish();
}

fn main() {}
