use pvlc_runtime_web::WebRuntime;

pub fn inspect_decoder_authority(runtime: &WebRuntime) {
    let _authority = &runtime.decoder_kv_session;
}
