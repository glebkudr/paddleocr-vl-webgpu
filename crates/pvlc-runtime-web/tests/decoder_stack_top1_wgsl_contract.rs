use naga::valid::{Capabilities, ValidationFlags, Validator};

const DECODER_STACK_SOURCE: &str = include_str!("../src/web/decoder_stack_session.rs");
const TOP1_START: &str = "const TOP1_SHADER_SOURCE: &str = r#\"";
const TOP1_END: &str = "\"#;\n\npub(super) struct BrowserDecoderStackResidentWeights";

fn top1_source() -> &'static str {
    let start = DECODER_STACK_SOURCE
        .find(TOP1_START)
        .expect("decoder GPU top-1 WGSL start marker is missing")
        + TOP1_START.len();
    let tail = &DECODER_STACK_SOURCE[start..];
    let end = tail
        .find(TOP1_END)
        .expect("decoder GPU top-1 WGSL end marker is missing");
    &tail[..end]
}

#[test]
fn decoder_gpu_top1_shader_parses_and_validates() {
    let source = top1_source();
    let module = naga::front::wgsl::parse_str(source).expect("decoder GPU top-1 WGSL must parse");
    Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .expect("decoder GPU top-1 WGSL must typecheck");

    assert_eq!(module.entry_points.len(), 1);
    let entry = &module.entry_points[0];
    assert_eq!(entry.name, "main");
    assert_eq!(entry.stage, naga::ShaderStage::Compute);
    assert_eq!(entry.workgroup_size, [256, 1, 1]);
}
