use pvlc_runtime_native::DecoderKvSessionDescriptor;
use pvlc_runtime_native::{NativeDecoderKvSession, NativeOptions, NativeRuntime};

fn escape() -> NativeDecoderKvSession<'static> {
    let runtime = NativeRuntime::new(NativeOptions::default()).unwrap();
    let keys = vec![0.0_f32; 4 * 2 * 128];
    let values = keys.clone();
    let descriptor = DecoderKvSessionDescriptor {
        query_heads: 16,
        key_value_heads: 2,
        head_dim: 128,
        prefix_tokens: 1,
        cache_capacity: 4,
        key_cache: &keys,
        value_cache: &values,
    };
    runtime.begin_decoder_kv_session(&descriptor).unwrap()
}

fn main() {}
