use std::any::TypeId;

use pvlc_ir::{
    PlanBinding, PlanBindingAccess, PlanBindingResource, PlanBufferId, PlanDtype, PlanError,
    PlanErrorCode, PlanExternalValue, PlanFeature, PlanIr, PlanNode, PlanNodeId, PlanOutput,
    PlanOutputBuffer, PlanRequirements, PlanTensorResource, PlanUniformResource, PlanValueId,
    SemanticId,
};
use pvlc_runtime_core::{InvocationPlan, KernelId};

const EXPECTED_CANONICAL_LEN: usize = 1_747;
const EXPECTED_CANONICAL_BLAKE3: &str =
    "e4f4bd110422d1d60b66df8be4d6ec8a66c566e04743783bebb80375cb388a42";
const EXPECTED_CANONICAL: &[u8] = br#"{"schema_version":1,"external_values":[{"id":"value.activation","dtype":"f32","shape":[3,3],"buffer_id":"buffer.activation","byte_offset":0,"byte_length":36}],"nodes":[{"id":"node.query","invocation":{"kernel":"vision_patch_projection_f32","output_elements":15,"output_bytes":60,"workgroup_size":[8,8,1],"dispatch":[1,1,1]},"uniform_words":[3,3,5,0],"bindings":[{"number":0,"access":"read_only_storage","resource":{"kind":"value","value_id":"value.activation"}},{"number":1,"access":"read_only_storage","resource":{"kind":"tensor","physical_name":"fixture.q.weight","semantic_id":"vision.layer.00.attention.q.weight","dtype":"bf16","shape":[5,3],"storage_format":"f32","buffer_id":"buffer.q_weight","byte_offset":0,"byte_length":60}},{"number":2,"access":"read_only_storage","resource":{"kind":"tensor","physical_name":"fixture.q.bias","semantic_id":"vision.layer.00.attention.q.bias","dtype":"bf16","shape":[5],"storage_format":"f32","buffer_id":"buffer.q_bias","byte_offset":0,"byte_length":20}},{"number":3,"access":"read_write_storage","resource":{"kind":"output_buffer","buffer_id":"buffer.query","byte_length":60}},{"number":4,"access":"uniform","resource":{"kind":"uniform_words","words":[3,3,5,0]}}],"outputs":[{"id":"value.query","dtype":"f32","shape":[3,5],"buffer_id":"buffer.query","byte_offset":0,"byte_length":60}],"diagnostic_label":"vision.layer.00.query","timestamp_label":null,"source_semantic_ids":["vision.layer.00.qkv"],"rewrite_provenance":null}],"outputs":["value.query"],"requirements":{"storage_binding_count":4,"uniform_binding_count":1,"required_storage_buffer_offset_alignment":32,"largest_storage_binding_bytes":60,"largest_buffer_bytes":60,"max_workgroup_size":[8,8,1],"max_dispatch":[1,1,1],"required_features":[]}}
"#;

fn node_id(value: &str) -> PlanNodeId {
    PlanNodeId::parse(value)
        .unwrap_or_else(|error| panic!("invalid test node ID {value:?}: {error}"))
}

fn value_id(value: &str) -> PlanValueId {
    PlanValueId::parse(value)
        .unwrap_or_else(|error| panic!("invalid test value ID {value:?}: {error}"))
}

fn buffer_id(value: &str) -> PlanBufferId {
    PlanBufferId::parse(value)
        .unwrap_or_else(|error| panic!("invalid test buffer ID {value:?}: {error}"))
}

fn semantic_id(value: &str) -> SemanticId {
    SemanticId::parse(value)
        .unwrap_or_else(|error| panic!("invalid test semantic ID {value:?}: {error}"))
}

fn tensor_binding(
    number: u32,
    physical_name: &str,
    semantic: &str,
    shape: &[u64],
    buffer: &str,
    byte_length: u64,
) -> PlanBinding {
    PlanBinding {
        number,
        access: PlanBindingAccess::ReadOnlyStorage,
        resource: PlanBindingResource::Tensor(PlanTensorResource {
            physical_name: physical_name.to_owned(),
            semantic_id: semantic_id(semantic),
            dtype: PlanDtype::BFloat16,
            shape: shape.to_vec(),
            storage_format: PlanDtype::Float32,
            buffer_id: buffer_id(buffer),
            byte_offset: 0,
            byte_length,
        }),
    }
}

fn projection_node(role: &str) -> PlanNode {
    let (weight_semantic, bias_semantic, weight_buffer, bias_buffer, output_buffer, output_value) =
        match role {
            "query" => (
                "vision.layer.00.attention.q.weight",
                "vision.layer.00.attention.q.bias",
                "buffer.q_weight",
                "buffer.q_bias",
                "buffer.query",
                "value.query",
            ),
            "key" => (
                "vision.layer.00.attention.k.weight",
                "vision.layer.00.attention.k.bias",
                "buffer.k_weight",
                "buffer.k_bias",
                "buffer.key",
                "value.key",
            ),
            other => panic!("unsupported test projection role {other}"),
        };
    PlanNode {
        id: node_id(&format!("node.{role}")),
        invocation: InvocationPlan {
            kernel: KernelId::VisionPatchProjectionF32,
            output_elements: 15,
            output_bytes: 60,
            workgroup_size: [8, 8, 1],
            dispatch: [1, 1, 1],
        },
        uniform_words: [3, 3, 5, 0],
        bindings: vec![
            PlanBinding {
                number: 0,
                access: PlanBindingAccess::ReadOnlyStorage,
                resource: PlanBindingResource::Value(value_id("value.activation")),
            },
            tensor_binding(
                1,
                &format!("fixture.{}.weight", &role[..1]),
                weight_semantic,
                &[5, 3],
                weight_buffer,
                60,
            ),
            tensor_binding(
                2,
                &format!("fixture.{}.bias", &role[..1]),
                bias_semantic,
                &[5],
                bias_buffer,
                20,
            ),
            PlanBinding {
                number: 3,
                access: PlanBindingAccess::ReadWriteStorage,
                resource: PlanBindingResource::OutputBuffer(PlanOutputBuffer {
                    buffer_id: buffer_id(output_buffer),
                    byte_length: 60,
                }),
            },
            PlanBinding {
                number: 4,
                access: PlanBindingAccess::Uniform,
                resource: PlanBindingResource::UniformWords(PlanUniformResource {
                    words: [3, 3, 5, 0],
                }),
            },
        ],
        outputs: vec![PlanOutput {
            id: value_id(output_value),
            dtype: PlanDtype::Float32,
            shape: vec![3, 5],
            buffer_id: buffer_id(output_buffer),
            byte_offset: 0,
            byte_length: 60,
        }],
        diagnostic_label: format!("vision.layer.00.{role}"),
        timestamp_label: None,
        source_semantic_ids: vec![semantic_id("vision.layer.00.qkv")],
        rewrite_provenance: None,
    }
}

fn requirements() -> PlanRequirements {
    PlanRequirements {
        storage_binding_count: 4,
        uniform_binding_count: 1,
        required_storage_buffer_offset_alignment: 32,
        largest_storage_binding_bytes: 60,
        largest_buffer_bytes: 60,
        max_workgroup_size: [8, 8, 1],
        max_dispatch: [1, 1, 1],
        required_features: vec![],
    }
}

fn fixture() -> PlanIr {
    PlanIr {
        schema_version: 1,
        external_values: vec![PlanExternalValue {
            id: value_id("value.activation"),
            dtype: PlanDtype::Float32,
            shape: vec![3, 3],
            buffer_id: buffer_id("buffer.activation"),
            byte_offset: 0,
            byte_length: 36,
        }],
        nodes: vec![projection_node("query")],
        outputs: vec![value_id("value.query")],
        requirements: requirements(),
    }
}

fn two_node_fixture() -> PlanIr {
    let mut plan = fixture();
    plan.nodes.push(projection_node("key"));
    plan.outputs.push(value_id("value.key"));
    plan
}

fn assert_verify_error(plan: &PlanIr, expected: PlanErrorCode) {
    let error = plan
        .verify()
        .expect_err("invalid PlanIR mutant must fail structural verification");
    assert_eq!(error.code(), expected, "unexpected error: {error}");
}

fn assert_verify_error_name(plan: &PlanIr, expected: &str) {
    let error = plan.verify().expect_err(
        "invalid PlanIR mutant must fail structural verification with the named stable code",
    );
    assert_eq!(
        format!("{:?}", error.code()),
        expected,
        "unexpected error: {error}"
    );
}

fn verify_error_name(plan: &PlanIr) -> Option<String> {
    plan.verify()
        .err()
        .map(|error| format!("{:?}", error.code()))
}

#[test]
fn plan_identifiers_are_distinct_validated_types() {
    assert_ne!(TypeId::of::<PlanNodeId>(), TypeId::of::<PlanValueId>());
    assert_ne!(TypeId::of::<PlanNodeId>(), TypeId::of::<PlanBufferId>());
    assert_ne!(TypeId::of::<PlanValueId>(), TypeId::of::<PlanBufferId>());
    assert_ne!(TypeId::of::<PlanNodeId>(), TypeId::of::<SemanticId>());
    assert_ne!(TypeId::of::<PlanValueId>(), TypeId::of::<SemanticId>());
    assert_ne!(TypeId::of::<PlanBufferId>(), TypeId::of::<SemanticId>());

    for valid in ["node.query", "vision.layer.00.query", "sentinel.before"] {
        assert_eq!(node_id(valid).as_str(), valid);
    }
    for valid in ["value.activation", "vision.layer.26.value"] {
        assert_eq!(value_id(valid).as_str(), valid);
    }
    for valid in ["buffer.activation", "weights.layer.00.query"] {
        assert_eq!(buffer_id(valid).as_str(), valid);
    }
    for invalid in [
        "",
        ".node",
        "node.",
        "node..query",
        "Node.query",
        "node/query",
        "node layer",
        "node.layer.0",
        "node.layer.000",
    ] {
        assert!(PlanNodeId::parse(invalid).is_err(), "accepted {invalid:?}");
        assert!(PlanValueId::parse(invalid).is_err(), "accepted {invalid:?}");
        assert!(
            PlanBufferId::parse(invalid).is_err(),
            "accepted {invalid:?}"
        );
    }
}

#[test]
fn canonical_json_parse_roundtrip_and_blake3_are_exact() {
    let plan = fixture();
    plan.verify().expect("independent fixture must verify");
    let actual = plan
        .canonical_bytes()
        .expect("verified PlanIR must serialize");

    assert_eq!(actual, EXPECTED_CANONICAL);
    assert_eq!(actual.len(), EXPECTED_CANONICAL_LEN);
    assert_eq!(actual.last(), Some(&b'\n'));
    assert_eq!(actual.iter().filter(|byte| **byte == b'\n').count(), 1);
    assert_eq!(
        plan.canonical_blake3_hex()
            .expect("verified PlanIR must hash"),
        EXPECTED_CANONICAL_BLAKE3
    );
    assert_eq!(
        blake3::hash(&actual).to_hex().as_str(),
        EXPECTED_CANONICAL_BLAKE3
    );

    let parsed =
        PlanIr::parse_canonical(EXPECTED_CANONICAL).expect("literal canonical PlanIR must parse");
    assert_eq!(parsed, plan);
    assert_eq!(parsed.canonical_bytes().unwrap(), EXPECTED_CANONICAL);
    assert_eq!(
        parsed.canonical_blake3_hex().unwrap(),
        EXPECTED_CANONICAL_BLAKE3
    );
}

#[test]
fn strict_parser_rejects_unknown_fields_and_noncanonical_encodings() {
    let canonical = std::str::from_utf8(EXPECTED_CANONICAL).unwrap();

    let mut unknown = canonical.strip_suffix("}\n").unwrap().to_owned();
    unknown.push_str(",\"unknown\":true}\n");
    let error = PlanIr::parse_canonical(unknown.as_bytes()).unwrap_err();
    assert_eq!(error.code(), PlanErrorCode::UnknownField);

    for noncanonical in [
        canonical.trim_end_matches('\n').to_owned(),
        format!(" {canonical}"),
        format!("{canonical}\n"),
    ] {
        let error = PlanIr::parse_canonical(noncanonical.as_bytes()).unwrap_err();
        assert_eq!(error.code(), PlanErrorCode::NonCanonicalEncoding);
    }

    let fixed_prefix = "{\"schema_version\":1,";
    let nodes_at = canonical.find(",\"nodes\":").unwrap();
    let external = &canonical[fixed_prefix.len()..nodes_at];
    let remainder = &canonical[nodes_at + 1..];
    let reordered = format!("{{{external},\"schema_version\":1,{remainder}");
    let error = PlanIr::parse_canonical(reordered.as_bytes()).unwrap_err();
    assert_eq!(error.code(), PlanErrorCode::NonCanonicalEncoding);

    for (scope, needle) in [
        ("node", "\"id\":\"node.query\","),
        ("invocation", "\"kernel\":\"vision_patch_projection_f32\","),
        ("binding", "\"number\":0,"),
        ("binding resource", "\"kind\":\"value\","),
        ("output", "\"id\":\"value.query\","),
        ("requirements", "\"storage_binding_count\":4,"),
    ] {
        let position = canonical
            .find(needle)
            .unwrap_or_else(|| panic!("missing {scope} injection point"))
            + needle.len();
        let mut nested_unknown = canonical.to_owned();
        nested_unknown.insert_str(position, "\"unknown\":true,");
        let error: PlanError = match PlanIr::parse_canonical(nested_unknown.as_bytes()) {
            Ok(_) => panic!("accepted unknown field inside {scope}"),
            Err(error) => error,
        };
        assert_eq!(error.code(), PlanErrorCode::UnknownField, "{scope}");
    }
}

#[test]
fn duplicate_ids_producers_bindings_and_dangling_values_fail_closed() {
    let mut wrong_schema = fixture();
    wrong_schema.schema_version = 2;
    assert_verify_error(&wrong_schema, PlanErrorCode::UnsupportedSchemaVersion);

    let mut duplicate_node = two_node_fixture();
    duplicate_node.nodes[1].id = duplicate_node.nodes[0].id.clone();
    assert_verify_error(&duplicate_node, PlanErrorCode::DuplicateNodeId);

    let mut duplicate_external = fixture();
    duplicate_external
        .external_values
        .push(duplicate_external.external_values[0].clone());
    assert_verify_error(&duplicate_external, PlanErrorCode::DuplicateValueProducer);

    let mut duplicate_producer = two_node_fixture();
    duplicate_producer.nodes[1].outputs[0].id = value_id("value.query");
    assert_verify_error(&duplicate_producer, PlanErrorCode::DuplicateValueProducer);

    let mut duplicate_binding = fixture();
    duplicate_binding.nodes[0].bindings[1].number = 0;
    assert_verify_error(&duplicate_binding, PlanErrorCode::DuplicateBindingNumber);

    let mut binding_kind_mismatch = fixture();
    binding_kind_mismatch.nodes[0].bindings[0].access = PlanBindingAccess::Uniform;
    assert_verify_error(
        &binding_kind_mismatch,
        PlanErrorCode::BindingResourceMismatch,
    );

    let mut dangling_input = fixture();
    dangling_input.nodes[0].bindings[0].resource =
        PlanBindingResource::Value(value_id("value.missing"));
    assert_verify_error(&dangling_input, PlanErrorCode::DanglingInputValue);

    let mut dangling_output = fixture();
    dangling_output.outputs[0] = value_id("value.missing");
    assert_verify_error(&dangling_output, PlanErrorCode::OutputWithoutProducer);
}

#[test]
fn topology_ordering_and_source_provenance_are_strict() {
    let mut self_input = fixture();
    self_input.nodes[0].bindings[0].resource = PlanBindingResource::Value(value_id("value.query"));
    assert_verify_error(&self_input, PlanErrorCode::InvalidTopology);

    let mut later_input = two_node_fixture();
    later_input.nodes[0].bindings[0].resource = PlanBindingResource::Value(value_id("value.key"));
    assert_verify_error(&later_input, PlanErrorCode::InvalidTopology);

    let mut empty_source = fixture();
    empty_source.nodes[0].source_semantic_ids.clear();
    assert_verify_error(&empty_source, PlanErrorCode::EmptySourceProvenance);

    let mut duplicate_source = fixture();
    duplicate_source.nodes[0]
        .source_semantic_ids
        .push(semantic_id("vision.layer.00.qkv"));
    assert_verify_error(&duplicate_source, PlanErrorCode::DuplicateSourceProvenance);

    let mut noncanonical_bindings = fixture();
    noncanonical_bindings.nodes[0].bindings.swap(1, 2);
    assert_verify_error(&noncanonical_bindings, PlanErrorCode::NonCanonicalOrder);
}

#[test]
fn zero_overflowed_and_inconsistent_shapes_and_ranges_are_rejected() {
    let mut zero_shape = fixture();
    zero_shape.nodes[0].outputs[0].shape[1] = 0;
    assert_verify_error(&zero_shape, PlanErrorCode::ZeroShapeDimension);

    let mut shape_overflow = fixture();
    shape_overflow.nodes[0].outputs[0].shape = vec![u64::MAX, 2];
    assert_verify_error(&shape_overflow, PlanErrorCode::ArithmeticOverflow);

    let mut range_overflow = fixture();
    range_overflow.nodes[0].outputs[0].byte_offset = u64::MAX;
    range_overflow.nodes[0].outputs[0].byte_length = 4;
    assert_verify_error(&range_overflow, PlanErrorCode::ByteRangeOverflow);

    let mut out_of_bounds = fixture();
    out_of_bounds.nodes[0].outputs[0].byte_offset = 4;
    assert_verify_error(&out_of_bounds, PlanErrorCode::SliceOutOfBounds);

    let mut overlapping = fixture();
    let mut second = overlapping.nodes[0].outputs[0].clone();
    second.id = value_id("value.query_overlap");
    second.byte_offset = 4;
    second.byte_length = 56;
    overlapping.nodes[0].outputs.push(second);
    overlapping.outputs.push(value_id("value.query_overlap"));
    assert_verify_error(&overlapping, PlanErrorCode::OverlappingOutputSlices);

    let mut wrong_output_elements = fixture();
    wrong_output_elements.nodes[0].invocation.output_elements = 14;
    assert_verify_error(
        &wrong_output_elements,
        PlanErrorCode::InvocationResourceMismatch,
    );

    let mut wrong_output_bytes = fixture();
    wrong_output_bytes.nodes[0].invocation.output_bytes = 64;
    assert_verify_error(
        &wrong_output_bytes,
        PlanErrorCode::InvocationResourceMismatch,
    );

    let mut wrong_uniform_resource = fixture();
    let PlanBindingResource::UniformWords(uniform) =
        &mut wrong_uniform_resource.nodes[0].bindings[4].resource
    else {
        panic!("fixture binding 4 must be uniform words");
    };
    uniform.words[3] = 1;
    assert_verify_error(
        &wrong_uniform_resource,
        PlanErrorCode::InvocationResourceMismatch,
    );

    let mut zero_dispatch = fixture();
    zero_dispatch.nodes[0].invocation.dispatch[0] = 0;
    assert_verify_error(&zero_dispatch, PlanErrorCode::InvocationResourceMismatch);
}

#[test]
fn f32_external_and_output_byte_lengths_are_exact() {
    let mutants = [
        ("zero external", {
            let mut plan = fixture();
            plan.external_values[0].byte_length = 0;
            plan
        }),
        ("short external", {
            let mut plan = fixture();
            plan.external_values[0].byte_length = 32;
            plan
        }),
        ("zero output", {
            let mut plan = fixture();
            plan.nodes[0].outputs[0].byte_length = 0;
            plan
        }),
        ("short output", {
            let mut plan = fixture();
            plan.nodes[0].outputs[0].byte_length = 56;
            plan
        }),
    ];
    assert!(
        mutants
            .iter()
            .all(|(_, plan)| plan.requirements == requirements())
    );
    let observed = mutants
        .iter()
        .map(|(name, plan)| (*name, verify_error_name(plan)))
        .collect::<Vec<_>>();
    let expected = mutants
        .iter()
        .map(|(name, _)| (*name, Some("ValueByteLengthMismatch".to_owned())))
        .collect::<Vec<_>>();
    assert_eq!(observed, expected);
}

#[test]
fn invocation_output_dtype_is_f32() {
    let mut wrong_dtype = fixture();
    wrong_dtype.nodes[0].outputs[0].dtype = PlanDtype::Float16;
    wrong_dtype.nodes[0].outputs[0].byte_length = 30;
    assert_verify_error_name(&wrong_dtype, "InvocationOutputDtypeMismatch");
}

#[test]
fn duplicate_plan_outputs_are_rejected() {
    let mut duplicate_export = fixture();
    duplicate_export.outputs.push(value_id("value.query"));
    assert_verify_error_name(&duplicate_export, "DuplicatePlanOutput");
}

#[test]
fn cross_node_parameter_ranges_may_not_overlap() {
    let mut overlapping_parameters = two_node_fixture();
    let PlanBindingResource::Tensor(query_weight) =
        &overlapping_parameters.nodes[0].bindings[1].resource
    else {
        panic!("query binding 1 must be a tensor");
    };
    let shared_parameter_buffer = query_weight.buffer_id.clone();
    let PlanBindingResource::Tensor(key_weight) =
        &mut overlapping_parameters.nodes[1].bindings[1].resource
    else {
        panic!("key binding 1 must be a tensor");
    };
    key_weight.buffer_id = shared_parameter_buffer;
    key_weight.byte_offset = 32;
    overlapping_parameters.requirements.largest_buffer_bytes = 92;
    assert_verify_error_name(&overlapping_parameters, "OverlappingParameterSlices");
}

#[test]
fn cross_node_output_ranges_may_not_overlap() {
    let mut overlapping_outputs = two_node_fixture();
    let shared_output_buffer = overlapping_outputs.nodes[0].outputs[0].buffer_id.clone();
    overlapping_outputs.nodes[1].outputs[0].buffer_id = shared_output_buffer.clone();
    let PlanBindingResource::OutputBuffer(key_output_binding) =
        &mut overlapping_outputs.nodes[1].bindings[3].resource
    else {
        panic!("key binding 3 must be an output buffer");
    };
    key_output_binding.buffer_id = shared_output_buffer;
    assert_verify_error(&overlapping_outputs, PlanErrorCode::OverlappingOutputSlices);
}

#[test]
fn input_parameter_and_output_storage_may_not_alias() {
    let mut input_output = fixture();
    input_output.nodes[0].outputs[0].buffer_id = buffer_id("buffer.activation");
    let PlanBindingResource::OutputBuffer(output) = &mut input_output.nodes[0].bindings[3].resource
    else {
        panic!("fixture binding 3 must be the output buffer");
    };
    output.buffer_id = buffer_id("buffer.activation");
    assert_verify_error(&input_output, PlanErrorCode::IllegalBufferAlias);

    let mut parameter_output = fixture();
    let PlanBindingResource::Tensor(weight) = &mut parameter_output.nodes[0].bindings[1].resource
    else {
        panic!("fixture binding 1 must be a tensor");
    };
    weight.buffer_id = buffer_id("buffer.query");
    assert_verify_error(&parameter_output, PlanErrorCode::IllegalBufferAlias);

    let mut input_parameter = fixture();
    let PlanBindingResource::Tensor(weight) = &mut input_parameter.nodes[0].bindings[1].resource
    else {
        panic!("fixture binding 1 must be a tensor");
    };
    weight.buffer_id = buffer_id("buffer.activation");
    assert_verify_error(&input_parameter, PlanErrorCode::IllegalBufferAlias);
}

#[test]
fn requirements_are_derived_only_from_selected_invocations_and_resources() {
    let plan = fixture();
    let derived = PlanRequirements::derive(
        &plan.external_values,
        &plan.nodes,
        32,
        &[] as &[PlanFeature],
    )
    .expect("compact requirements must derive");
    assert_eq!(derived, requirements());

    let mut two_nodes = two_node_fixture();
    two_nodes.requirements =
        PlanRequirements::derive(&two_nodes.external_values, &two_nodes.nodes, 32, &[]).unwrap();
    assert_eq!(two_nodes.requirements, requirements());

    let mut corrupted = fixture();
    corrupted.requirements.largest_buffer_bytes += 1;
    assert_verify_error(&corrupted, PlanErrorCode::RequirementsMismatch);
}

#[test]
fn required_features_have_stable_names_order_and_derive_without_loss() {
    assert_eq!(PlanFeature::ShaderF16.as_str(), "shader_f16");
    assert_eq!(PlanFeature::TimestampQuery.as_str(), "timestamp_query");
    let ordered = [PlanFeature::ShaderF16, PlanFeature::TimestampQuery];

    let mut plan = fixture();
    plan.requirements = PlanRequirements::derive(&plan.external_values, &plan.nodes, 32, &ordered)
        .expect("ordered non-empty feature requirements must derive");
    assert_eq!(plan.requirements.required_features, ordered);
    plan.verify()
        .expect("derived feature requirements must verify");
    let bytes = plan.canonical_bytes().unwrap();
    assert!(
        std::str::from_utf8(&bytes)
            .unwrap()
            .contains("\"required_features\":[\"shader_f16\",\"timestamp_query\"]")
    );
    assert_eq!(
        PlanIr::parse_canonical(&bytes)
            .unwrap()
            .requirements
            .required_features,
        ordered
    );

    let duplicate = PlanRequirements::derive(
        &plan.external_values,
        &plan.nodes,
        32,
        &[PlanFeature::ShaderF16, PlanFeature::ShaderF16],
    )
    .unwrap_err();
    assert_eq!(duplicate.code(), PlanErrorCode::DuplicateRequiredFeature);

    let reordered = PlanRequirements::derive(
        &plan.external_values,
        &plan.nodes,
        32,
        &[PlanFeature::TimestampQuery, PlanFeature::ShaderF16],
    )
    .unwrap_err();
    assert_eq!(reordered.code(), PlanErrorCode::NonCanonicalOrder);
}
