#![cfg(target_arch = "wasm32")]

use js_sys::Uint8Array;
use pvlc_runtime_web::WebRuntime;

fn assert_full_layer_session_surface(
    runtime: &WebRuntime,
    descriptor_json: &str,
    bytes: &Uint8Array,
    source: &str,
) {
    let _creation = runtime.begin_decoder_full_layer_session(descriptor_json, bytes, bytes, bytes);
    let _override_creation = runtime.begin_decoder_full_layer_session_with_shader_override(
        descriptor_json,
        bytes,
        bytes,
        bytes,
        "decoder_swiglu_f32",
        source,
    );
    let _step = runtime.step_decoder_full_layer_session(bytes);
    let _finish = runtime.finish_decoder_full_layer_session();
    runtime.abort_decoder_full_layer_session();
    let _sources: Result<String, wasm_bindgen::JsValue> =
        runtime.decoder_full_layer_session_shader_sources_json();
}

#[test]
fn wasm_runtime_owns_the_complete_full_layer_session_operation_surface() {
    let _ = assert_full_layer_session_surface;
}
