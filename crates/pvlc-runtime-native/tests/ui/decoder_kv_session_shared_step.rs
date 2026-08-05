use pvlc_runtime_native::DecoderKvSessionStep;
use pvlc_runtime_native::NativeDecoderKvSession;

fn step_through_shared_reference(
    session: &NativeDecoderKvSession<'_>,
    step: &DecoderKvSessionStep<'_>,
) {
    let _ = session.step(step);
}

fn main() {}
