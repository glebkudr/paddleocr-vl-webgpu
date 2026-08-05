use pvlc_runtime_web::WebRuntime;

pub fn inspect_decoder_stack_authority(runtime: &WebRuntime) {
    let _authority = &runtime.decoder_stack_session;
}
