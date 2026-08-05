/* tslint:disable */
/* eslint-disable */

export class WebRuntime {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    abort_vision_encoder_stack_sharded(): void;
    begin_vision_encoder_stack_sharded_json(manifest_json: string): string;
    begin_vision_encoder_stack_sharded_with_activation_strategy_and_memory_hardening_and_qkv_selection_json(manifest_json: string, activation_strategy: string, memory_hardening: string, qkv_selection: WebVisionQkvStackSelection): string;
    begin_vision_encoder_stack_sharded_with_activation_strategy_and_memory_hardening_json(manifest_json: string, activation_strategy: string, memory_hardening: string): string;
    begin_vision_encoder_stack_sharded_with_activation_strategy_and_qkv_selection_json(manifest_json: string, activation_strategy: string, qkv_selection: WebVisionQkvStackSelection): string;
    begin_vision_encoder_stack_sharded_with_activation_strategy_json(manifest_json: string, activation_strategy: string): string;
    blake3_bytes_hex(bytes: Uint8Array): string;
    blake3_hex(source: string): string;
    capabilities_json(): string;
    compile_vision_encoder_stack_qkv_selection(manifest_json: string, policy: string): WebVisionQkvStackSelection;
    static create(): Promise<WebRuntime>;
    finish_vision_encoder_stack_sharded(shard_id: string, bytes: Uint8Array): Promise<any>;
    preflight_vision_encoder_stack_shard_json(shard_id: string, bytes: Uint8Array): string;
    probe_validation_error_json(label: string, source: string, missing_entry_point: string): Promise<string>;
    projector_shader_sources_json(): string;
    run_json(invocation_json: string): Promise<string>;
    run_projector_bytes(descriptor_json: string, profile: string, input: Uint8Array, weights: Uint8Array): Promise<any>;
    run_projector_json(invocation_json: string, readback: string): Promise<string>;
    run_projector_with_shader_override_json(invocation_json: string, readback: string, kernel: string, source: string): Promise<string>;
    run_vision_encoder_layer_identity_rope_bytes(descriptor_json: string, weights: Uint8Array, readback: string): Promise<any>;
    run_vision_encoder_layer_identity_rope_json(invocation_json: string, readback: string): Promise<string>;
    run_vision_encoder_layer_identity_rope_with_shader_override_json(invocation_json: string, readback: string, kernel: string, source: string): Promise<string>;
    run_vision_encoder_stack_sharded_layer_json(shard_id: string, bytes: Uint8Array): Promise<string>;
    run_with_shader_json(invocation_json: string, label: string, source: string, entry_point: string): Promise<string>;
    start_vision_encoder_stack_sharded_json(shard_id: string, bytes: Uint8Array): Promise<string>;
    validate_all_pipelines_json(): Promise<string>;
    validate_projector_pipelines_json(): Promise<string>;
    validate_vision_attention_pipeline_json(): Promise<string>;
    validate_vision_encoder_layer_pipelines_json(): Promise<string>;
    vision_encoder_layer_shader_sources_json(): string;
    vision_encoder_stack_qkv_shader_sources_json(activation_strategy: string): string;
    vision_encoder_stack_shader_sources_json(activation_strategy: string): string;
}

export class WebVisionQkvStackSelection {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    evidence_json(): string;
}

export function assemble_browser_benchmark_cohort_v1(canonical_input: Uint8Array): Uint8Array;

export function canonical_vision_encoder_stack_shader_sources_json(activation_strategy: string): string;

export function validate_browser_benchmark_cohort_plan_v1(canonical_plan: Uint8Array): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_webruntime_free: (a: number, b: number) => void;
    readonly __wbg_webvisionqkvstackselection_free: (a: number, b: number) => void;
    readonly assemble_browser_benchmark_cohort_v1: (a: any) => [number, number, number];
    readonly canonical_vision_encoder_stack_shader_sources_json: (a: number, b: number) => [number, number, number, number];
    readonly validate_browser_benchmark_cohort_plan_v1: (a: any) => [number, number];
    readonly webruntime_abort_vision_encoder_stack_sharded: (a: number) => void;
    readonly webruntime_begin_vision_encoder_stack_sharded_json: (a: number, b: number, c: number) => [number, number, number, number];
    readonly webruntime_begin_vision_encoder_stack_sharded_with_activation_strategy_and_memory_hardening_and_qkv_selection_json: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number, number];
    readonly webruntime_begin_vision_encoder_stack_sharded_with_activation_strategy_and_memory_hardening_json: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number, number];
    readonly webruntime_begin_vision_encoder_stack_sharded_with_activation_strategy_and_qkv_selection_json: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly webruntime_begin_vision_encoder_stack_sharded_with_activation_strategy_json: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly webruntime_blake3_bytes_hex: (a: number, b: any) => [number, number];
    readonly webruntime_blake3_hex: (a: number, b: number, c: number) => [number, number];
    readonly webruntime_capabilities_json: (a: number) => [number, number, number, number];
    readonly webruntime_compile_vision_encoder_stack_qkv_selection: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly webruntime_create: () => any;
    readonly webruntime_finish_vision_encoder_stack_sharded: (a: number, b: number, c: number, d: any) => any;
    readonly webruntime_preflight_vision_encoder_stack_shard_json: (a: number, b: number, c: number, d: any) => [number, number, number, number];
    readonly webruntime_probe_validation_error_json: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => any;
    readonly webruntime_projector_shader_sources_json: (a: number) => [number, number, number, number];
    readonly webruntime_run_json: (a: number, b: number, c: number) => any;
    readonly webruntime_run_projector_bytes: (a: number, b: number, c: number, d: number, e: number, f: any, g: any) => any;
    readonly webruntime_run_projector_json: (a: number, b: number, c: number, d: number, e: number) => any;
    readonly webruntime_run_projector_with_shader_override_json: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => any;
    readonly webruntime_run_vision_encoder_layer_identity_rope_bytes: (a: number, b: number, c: number, d: any, e: number, f: number) => any;
    readonly webruntime_run_vision_encoder_layer_identity_rope_json: (a: number, b: number, c: number, d: number, e: number) => any;
    readonly webruntime_run_vision_encoder_layer_identity_rope_with_shader_override_json: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => any;
    readonly webruntime_run_vision_encoder_stack_sharded_layer_json: (a: number, b: number, c: number, d: any) => any;
    readonly webruntime_run_with_shader_json: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => any;
    readonly webruntime_start_vision_encoder_stack_sharded_json: (a: number, b: number, c: number, d: any) => any;
    readonly webruntime_validate_all_pipelines_json: (a: number) => any;
    readonly webruntime_validate_projector_pipelines_json: (a: number) => any;
    readonly webruntime_validate_vision_attention_pipeline_json: (a: number) => any;
    readonly webruntime_validate_vision_encoder_layer_pipelines_json: (a: number) => any;
    readonly webruntime_vision_encoder_layer_shader_sources_json: (a: number) => [number, number, number, number];
    readonly webruntime_vision_encoder_stack_qkv_shader_sources_json: (a: number, b: number, c: number) => [number, number, number, number];
    readonly webruntime_vision_encoder_stack_shader_sources_json: (a: number, b: number, c: number) => [number, number, number, number];
    readonly webvisionqkvstackselection_evidence_json: (a: number) => [number, number, number, number];
    readonly wasm_bindgen__convert__closures_____invoke__hecc1b4ac3e013480: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__hd6e68aeb014ef55b: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h78a66d65a0efada1: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h8114d0373a2d315f: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h3ab6f63f82218151: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h90f37b8adbec870a: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__hb70b2615af075870: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h9829432a97365cad: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__hb98af911e1a1d97c: (a: number, b: number) => number;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_destroy_closure: (a: number, b: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
