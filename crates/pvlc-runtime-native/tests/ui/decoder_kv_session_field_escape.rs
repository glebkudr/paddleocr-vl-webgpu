use pvlc_runtime_native::NativeDecoderKvSession;

fn expose(session: &NativeDecoderKvSession<'_>) {
    let _ = &session.key_cache_buffer;
}

fn main() {}
