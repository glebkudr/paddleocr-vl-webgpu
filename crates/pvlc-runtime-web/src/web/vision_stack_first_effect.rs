use super::{BrowserVisionStackError, FirstWebGpuEffectOutput, PreparedFirstWebGpuEffect};
use crate::vision_stack_causal::VisionStackPostEffectToken;
use wasm_bindgen::JsValue;

pub(super) fn run_first_webgpu_effect(
    device: &wgpu::Device,
    post_effect: &VisionStackPostEffectToken,
    effect: PreparedFirstWebGpuEffect<'_>,
) -> Result<FirstWebGpuEffectOutput, BrowserVisionStackError> {
    let _ = post_effect;
    match effect {
        PreparedFirstWebGpuEffect::PushErrorScope {
            raw_device,
            push,
            filter,
        } => match push.call1(raw_device, &JsValue::from_str(filter)) {
            Ok(_) => Ok(FirstWebGpuEffectOutput::ErrorScope),
            Err(_) => Err(BrowserVisionStackError(String::from(
                "cannot push the sealed first WebGPU error scope",
            ))),
        },
        PreparedFirstWebGpuEffect::CreateShaderModule { label, source } => {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(source)),
            });
            Ok(FirstWebGpuEffectOutput::ShaderModule(module))
        }
        PreparedFirstWebGpuEffect::CreateComputePipeline {
            label,
            module,
            entry_point,
        } => {
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: true,
                },
                cache: None,
            });
            Ok(FirstWebGpuEffectOutput::ComputePipeline(pipeline))
        }
    }
}
