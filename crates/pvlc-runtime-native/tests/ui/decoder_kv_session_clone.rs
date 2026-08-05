use pvlc_runtime_native::NativeDecoderKvSession;

fn require_clone<T: Clone>() {}

fn duplicate() {
    require_clone::<NativeDecoderKvSession<'static>>();
}

fn main() {}
