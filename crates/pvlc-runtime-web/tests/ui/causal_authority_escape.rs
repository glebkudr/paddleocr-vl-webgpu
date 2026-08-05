use pvlc_runtime_web::vision_stack_causal::{
    VisionStackErrorScopeAuthority, VisionStackErrorScopedOperation,
    VisionStackOperationEffectBoundary, VisionStackOperationEffectResult,
    VisionStackPostEffectToken,
};

fn forge(
    _boundary: VisionStackOperationEffectBoundary,
    _token: VisionStackPostEffectToken,
    _result: VisionStackOperationEffectResult<(), ()>,
    _scope_authority: VisionStackErrorScopeAuthority<'static, ()>,
    _scoped_operation: VisionStackErrorScopedOperation<(), (), ()>,
) {
}

fn main() {}
