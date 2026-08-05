use std::collections::{BTreeMap, BTreeSet};

use pvlc_model_schema::{
    COMPILER_MODEL_ABI, MODEL_ID, MODEL_REVISION, ObservedTensor, PaddleOcrVl16Schema,
    SchemaErrorCode, TensorDtype,
};

const EXPECTED_CATALOG_BLAKE3: &str =
    "422b0532712a71baee6d085de1fc79f057764f2fee71545c18a680d1b97543a6";
const EXPECTED_CATALOG_LEN: usize = 58_966;
const EXPECTED_SEMANTIC_MAP_BLAKE3: &str =
    "749a6e5c7d91013b13b4e77cb967a078015ba9b3f791e582f04c791beff086b7";
const EXPECTED_SEMANTIC_MAP_LEN: usize = 68_676;

#[derive(Clone, Debug, Eq, PartialEq)]
struct ExpectedTensor {
    dtype: &'static str,
    shape: Vec<u64>,
}

fn insert(catalog: &mut BTreeMap<String, ExpectedTensor>, name: impl Into<String>, shape: &[u64]) {
    let old = catalog.insert(
        name.into(),
        ExpectedTensor {
            dtype: "BF16",
            shape: shape.to_vec(),
        },
    );
    assert!(
        old.is_none(),
        "the independent expected catalog has a duplicate"
    );
}

/// Independently spells out the checkpoint families. This deliberately does not
/// call schema production helpers: changing a loop or shape in production must
/// still be caught here and by the pinned canonical hash.
fn expected_catalog() -> BTreeMap<String, ExpectedTensor> {
    let mut out = BTreeMap::new();

    insert(&mut out, "lm_head.weight", &[103_424, 1_024]);
    insert(&mut out, "mlp_AR.linear_1.bias", &[4_608]);
    insert(&mut out, "mlp_AR.linear_1.weight", &[4_608, 4_608]);
    insert(&mut out, "mlp_AR.linear_2.bias", &[1_024]);
    insert(&mut out, "mlp_AR.linear_2.weight", &[1_024, 4_608]);
    insert(&mut out, "mlp_AR.pre_norm.bias", &[1_152]);
    insert(&mut out, "mlp_AR.pre_norm.weight", &[1_152]);
    insert(&mut out, "model.embed_tokens.weight", &[103_424, 1_024]);
    insert(&mut out, "model.norm.weight", &[1_024]);

    for layer in 0..18 {
        let p = format!("model.layers.{layer}");
        insert(&mut out, format!("{p}.input_layernorm.weight"), &[1_024]);
        insert(
            &mut out,
            format!("{p}.post_attention_layernorm.weight"),
            &[1_024],
        );
        insert(
            &mut out,
            format!("{p}.self_attn.q_proj.weight"),
            &[2_048, 1_024],
        );
        insert(
            &mut out,
            format!("{p}.self_attn.k_proj.weight"),
            &[256, 1_024],
        );
        insert(
            &mut out,
            format!("{p}.self_attn.v_proj.weight"),
            &[256, 1_024],
        );
        insert(
            &mut out,
            format!("{p}.self_attn.o_proj.weight"),
            &[1_024, 2_048],
        );
        insert(
            &mut out,
            format!("{p}.mlp.gate_proj.weight"),
            &[3_072, 1_024],
        );
        insert(&mut out, format!("{p}.mlp.up_proj.weight"), &[3_072, 1_024]);
        insert(
            &mut out,
            format!("{p}.mlp.down_proj.weight"),
            &[1_024, 3_072],
        );
    }

    insert(
        &mut out,
        "visual.vision_model.embeddings.packing_position_embedding.weight",
        &[32_768, 1_152],
    );
    insert(
        &mut out,
        "visual.vision_model.embeddings.patch_embedding.bias",
        &[1_152],
    );
    insert(
        &mut out,
        "visual.vision_model.embeddings.patch_embedding.weight",
        &[1_152, 3, 14, 14],
    );
    insert(
        &mut out,
        "visual.vision_model.embeddings.position_embedding.weight",
        &[729, 1_152],
    );

    for layer in 0..27 {
        let p = format!("visual.vision_model.encoder.layers.{layer}");
        for norm in ["layer_norm1", "layer_norm2"] {
            insert(&mut out, format!("{p}.{norm}.bias"), &[1_152]);
            insert(&mut out, format!("{p}.{norm}.weight"), &[1_152]);
        }
        insert(&mut out, format!("{p}.mlp.fc1.bias"), &[4_304]);
        insert(&mut out, format!("{p}.mlp.fc1.weight"), &[4_304, 1_152]);
        insert(&mut out, format!("{p}.mlp.fc2.bias"), &[1_152]);
        insert(&mut out, format!("{p}.mlp.fc2.weight"), &[1_152, 4_304]);
        for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            insert(
                &mut out,
                format!("{p}.self_attn.{projection}.bias"),
                &[1_152],
            );
            insert(
                &mut out,
                format!("{p}.self_attn.{projection}.weight"),
                &[1_152, 1_152],
            );
        }
    }

    insert(
        &mut out,
        "visual.vision_model.head.attention.in_proj_bias",
        &[3_456],
    );
    insert(
        &mut out,
        "visual.vision_model.head.attention.in_proj_weight",
        &[3_456, 1_152],
    );
    insert(
        &mut out,
        "visual.vision_model.head.attention.out_proj.bias",
        &[1_152],
    );
    insert(
        &mut out,
        "visual.vision_model.head.attention.out_proj.weight",
        &[1_152, 1_152],
    );
    insert(
        &mut out,
        "visual.vision_model.head.layernorm.bias",
        &[1_152],
    );
    insert(
        &mut out,
        "visual.vision_model.head.layernorm.weight",
        &[1_152],
    );
    insert(&mut out, "visual.vision_model.head.mlp.fc1.bias", &[4_304]);
    insert(
        &mut out,
        "visual.vision_model.head.mlp.fc1.weight",
        &[4_304, 1_152],
    );
    insert(&mut out, "visual.vision_model.head.mlp.fc2.bias", &[1_152]);
    insert(
        &mut out,
        "visual.vision_model.head.mlp.fc2.weight",
        &[1_152, 4_304],
    );
    insert(&mut out, "visual.vision_model.head.probe", &[1, 1, 1_152]);
    insert(
        &mut out,
        "visual.vision_model.post_layernorm.bias",
        &[1_152],
    );
    insert(
        &mut out,
        "visual.vision_model.post_layernorm.weight",
        &[1_152],
    );

    assert_eq!(out.len(), 620);
    out
}

fn insert_role(
    roles: &mut BTreeMap<String, String>,
    name: impl Into<String>,
    semantic_id: impl Into<String>,
) {
    let old = roles.insert(name.into(), semantic_id.into());
    assert!(
        old.is_none(),
        "the independent semantic map has a duplicate"
    );
}

/// Full independent physical-name -> semantic-role contract. This is separate
/// from the shape fixture so neither production nor a test helper can make up
/// arbitrary unique IDs for the unchecked majority of the checkpoint.
fn expected_semantic_map() -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    insert_role(&mut out, "lm_head.weight", "lm_head.weight");
    for suffix in ["bias", "weight"] {
        insert_role(
            &mut out,
            format!("mlp_AR.pre_norm.{suffix}"),
            format!("projector.pre_norm.{suffix}"),
        );
        insert_role(
            &mut out,
            format!("mlp_AR.linear_1.{suffix}"),
            format!("projector.linear1.{suffix}"),
        );
        insert_role(
            &mut out,
            format!("mlp_AR.linear_2.{suffix}"),
            format!("projector.linear2.{suffix}"),
        );
    }
    insert_role(
        &mut out,
        "model.embed_tokens.weight",
        "decoder.embedding.weight",
    );
    insert_role(&mut out, "model.norm.weight", "decoder.final_norm.weight");

    for layer in 0..18 {
        let physical = format!("model.layers.{layer}");
        let semantic = format!("decoder.layer.{layer:02}");
        insert_role(
            &mut out,
            format!("{physical}.input_layernorm.weight"),
            format!("{semantic}.norm1.weight"),
        );
        insert_role(
            &mut out,
            format!("{physical}.post_attention_layernorm.weight"),
            format!("{semantic}.norm2.weight"),
        );
        for (physical_projection, semantic_projection) in [
            ("q_proj", "q"),
            ("k_proj", "k"),
            ("v_proj", "v"),
            ("o_proj", "out"),
        ] {
            insert_role(
                &mut out,
                format!("{physical}.self_attn.{physical_projection}.weight"),
                format!("{semantic}.attention.{semantic_projection}.weight"),
            );
        }
        for (physical_projection, semantic_projection) in [
            ("gate_proj", "gate"),
            ("up_proj", "up"),
            ("down_proj", "down"),
        ] {
            insert_role(
                &mut out,
                format!("{physical}.mlp.{physical_projection}.weight"),
                format!("{semantic}.mlp.{semantic_projection}.weight"),
            );
        }
    }

    insert_role(
        &mut out,
        "visual.vision_model.embeddings.packing_position_embedding.weight",
        "vision.embeddings.packing_position.weight",
    );
    insert_role(
        &mut out,
        "visual.vision_model.embeddings.patch_embedding.bias",
        "vision.embeddings.patch.bias",
    );
    insert_role(
        &mut out,
        "visual.vision_model.embeddings.patch_embedding.weight",
        "vision.embeddings.patch.weight",
    );
    insert_role(
        &mut out,
        "visual.vision_model.embeddings.position_embedding.weight",
        "vision.embeddings.position.weight",
    );

    for layer in 0..27 {
        let physical = format!("visual.vision_model.encoder.layers.{layer}");
        let semantic = format!("vision.layer.{layer:02}");
        for (physical_norm, semantic_norm) in [("layer_norm1", "norm1"), ("layer_norm2", "norm2")] {
            for suffix in ["bias", "weight"] {
                insert_role(
                    &mut out,
                    format!("{physical}.{physical_norm}.{suffix}"),
                    format!("{semantic}.{semantic_norm}.{suffix}"),
                );
            }
        }
        for projection in ["fc1", "fc2"] {
            for suffix in ["bias", "weight"] {
                insert_role(
                    &mut out,
                    format!("{physical}.mlp.{projection}.{suffix}"),
                    format!("{semantic}.mlp.{projection}.{suffix}"),
                );
            }
        }
        for (physical_projection, semantic_projection) in [
            ("q_proj", "q"),
            ("k_proj", "k"),
            ("v_proj", "v"),
            ("out_proj", "out"),
        ] {
            for suffix in ["bias", "weight"] {
                insert_role(
                    &mut out,
                    format!("{physical}.self_attn.{physical_projection}.{suffix}"),
                    format!("{semantic}.attention.{semantic_projection}.{suffix}"),
                );
            }
        }
    }

    for suffix in ["bias", "weight"] {
        insert_role(
            &mut out,
            format!("visual.vision_model.head.attention.in_proj_{suffix}"),
            format!("vision.head.attention.qkv.{suffix}"),
        );
        insert_role(
            &mut out,
            format!("visual.vision_model.head.attention.out_proj.{suffix}"),
            format!("vision.head.attention.out.{suffix}"),
        );
        insert_role(
            &mut out,
            format!("visual.vision_model.head.layernorm.{suffix}"),
            format!("vision.head.norm.{suffix}"),
        );
        for projection in ["fc1", "fc2"] {
            insert_role(
                &mut out,
                format!("visual.vision_model.head.mlp.{projection}.{suffix}"),
                format!("vision.head.mlp.{projection}.{suffix}"),
            );
        }
        insert_role(
            &mut out,
            format!("visual.vision_model.post_layernorm.{suffix}"),
            format!("vision.post_norm.{suffix}"),
        );
    }
    insert_role(
        &mut out,
        "visual.vision_model.head.probe",
        "vision.head.probe",
    );

    let catalog = expected_catalog();
    assert_eq!(out.len(), 620);
    assert_eq!(
        out.keys().collect::<Vec<_>>(),
        catalog.keys().collect::<Vec<_>>()
    );
    assert_eq!(out.values().collect::<BTreeSet<_>>().len(), 620);
    out
}

fn expected_canonical_bytes() -> Vec<u8> {
    let records: Vec<_> = expected_catalog()
        .into_iter()
        .map(|(name, tensor)| {
            serde_json::json!({
                "dtype": tensor.dtype,
                "name": name,
                "shape": tensor.shape,
            })
        })
        .collect();
    let mut bytes = serde_json::to_vec(&records).expect("serializing test fixture cannot fail");
    bytes.push(b'\n');
    bytes
}

fn canonical_semantic_map_bytes(roles: impl IntoIterator<Item = (String, String)>) -> Vec<u8> {
    let records: Vec<_> = roles
        .into_iter()
        .map(|(name, semantic_id)| {
            serde_json::json!({
                "name": name,
                "semantic_id": semantic_id,
            })
        })
        .collect();
    let mut bytes = serde_json::to_vec(&records).expect("serializing test fixture cannot fail");
    bytes.push(b'\n');
    bytes
}

#[test]
fn model_identity_is_a_compile_time_contract() {
    assert_eq!(MODEL_ID, "PaddlePaddle/PaddleOCR-VL-1.6");
    assert_eq!(MODEL_REVISION, "66317acc4c9fc17bd154591ce650735cd2855f3e");
    assert_eq!(COMPILER_MODEL_ABI, 1);
}

#[test]
fn exact_catalog_matches_all_620_checkpoint_tensors() {
    let actual = PaddleOcrVl16Schema::tensor_specs();
    let expected = expected_catalog();
    let expected_roles = expected_semantic_map();

    assert_eq!(actual.len(), 620);
    assert!(actual.windows(2).all(|w| w[0].name < w[1].name));
    assert_eq!(
        actual
            .iter()
            .map(|s| &s.name)
            .collect::<BTreeSet<_>>()
            .len(),
        actual.len(),
        "tensor names must be unique"
    );
    assert_eq!(
        actual
            .iter()
            .map(|s| s.semantic_id.as_str())
            .collect::<BTreeSet<_>>()
            .len(),
        actual.len(),
        "every physical tensor needs a unique stable semantic role"
    );

    for spec in &actual {
        let expected = expected
            .get(&spec.name)
            .unwrap_or_else(|| panic!("unexpected tensor {}", spec.name));
        assert_eq!(spec.dtype, TensorDtype::BFloat16, "{}", spec.name);
        assert_eq!(spec.shape, expected.shape, "{}", spec.name);
        assert_eq!(
            spec.semantic_id.as_str(),
            expected_roles[&spec.name],
            "{}",
            spec.name
        );
    }
}

#[test]
fn canonical_catalog_is_byte_stable_and_independently_anchored() {
    let actual = PaddleOcrVl16Schema::canonical_catalog_bytes();
    let expected = expected_canonical_bytes();

    assert_eq!(actual, expected);
    assert_eq!(actual.len(), EXPECTED_CATALOG_LEN);
    assert_eq!(
        blake3::hash(&actual).to_hex().as_str(),
        EXPECTED_CATALOG_BLAKE3
    );
    assert_eq!(actual.last(), Some(&b'\n'));
}

#[test]
fn complete_semantic_role_map_is_byte_stable_and_independently_anchored() {
    let expected = canonical_semantic_map_bytes(expected_semantic_map());
    let specs = PaddleOcrVl16Schema::tensor_specs();
    let actual = canonical_semantic_map_bytes(
        specs
            .iter()
            .map(|spec| (spec.name.clone(), spec.semantic_id.clone())),
    );

    assert_eq!(actual, expected);
    assert_eq!(
        PaddleOcrVl16Schema::canonical_semantic_map_bytes(),
        expected
    );
    assert_eq!(actual.len(), EXPECTED_SEMANTIC_MAP_LEN);
    assert_eq!(
        blake3::hash(&actual).to_hex().as_str(),
        EXPECTED_SEMANTIC_MAP_BLAKE3
    );
}

#[test]
fn asymmetric_and_boundary_shapes_cannot_be_silently_transposed() {
    let specs = PaddleOcrVl16Schema::tensor_specs();
    let get = |name: &str| {
        specs
            .iter()
            .find(|spec| spec.name == name)
            .unwrap_or_else(|| panic!("missing {name}"))
    };

    assert_eq!(
        get("visual.vision_model.embeddings.patch_embedding.weight").shape,
        [1_152, 3, 14, 14]
    );
    assert_eq!(
        get("visual.vision_model.encoder.layers.26.mlp.fc2.weight").shape,
        [1_152, 4_304]
    );
    assert_eq!(
        get("model.layers.0.self_attn.q_proj.weight").shape,
        [2_048, 1_024]
    );
    assert_eq!(
        get("model.layers.17.self_attn.k_proj.weight").shape,
        [256, 1_024]
    );
    assert_eq!(
        get("model.layers.17.self_attn.o_proj.weight").shape,
        [1_024, 2_048]
    );
    assert_eq!(get("mlp_AR.linear_1.weight").shape, [4_608, 4_608]);
    assert_eq!(get("lm_head.weight").shape, [103_424, 1_024]);
}

#[test]
fn selected_tensor_semantic_roles_are_stable() {
    let specs = PaddleOcrVl16Schema::tensor_specs();
    let role = |name: &str| {
        specs
            .iter()
            .find(|spec| spec.name == name)
            .unwrap_or_else(|| panic!("missing {name}"))
            .semantic_id
            .as_str()
    };

    assert_eq!(
        role("visual.vision_model.embeddings.patch_embedding.weight"),
        "vision.embeddings.patch.weight"
    );
    assert_eq!(
        role("visual.vision_model.encoder.layers.0.self_attn.q_proj.weight"),
        "vision.layer.00.attention.q.weight"
    );
    assert_eq!(
        role("visual.vision_model.encoder.layers.26.mlp.fc2.bias"),
        "vision.layer.26.mlp.fc2.bias"
    );
    assert_eq!(role("mlp_AR.linear_2.weight"), "projector.linear2.weight");
    assert_eq!(
        role("model.layers.17.self_attn.k_proj.weight"),
        "decoder.layer.17.attention.k.weight"
    );
    assert_eq!(
        role("model.layers.17.mlp.down_proj.weight"),
        "decoder.layer.17.mlp.down.weight"
    );
    assert_eq!(
        role("model.embed_tokens.weight"),
        "decoder.embedding.weight"
    );
    assert_eq!(role("lm_head.weight"), "lm_head.weight");
}

fn observed_exact_catalog() -> Vec<ObservedTensor> {
    expected_catalog()
        .into_iter()
        .map(|(name, expected)| ObservedTensor::new(name, TensorDtype::BFloat16, expected.shape))
        .collect()
}

#[test]
fn validator_accepts_only_the_complete_exact_catalog() {
    PaddleOcrVl16Schema::validate(&observed_exact_catalog()).expect("exact catalog must pass");
}

#[test]
fn validator_reports_an_unexpected_tensor_with_a_machine_readable_code() {
    let mut unexpected = observed_exact_catalog();
    unexpected.push(ObservedTensor::new(
        "model.layers.18.input_layernorm.weight",
        TensorDtype::BFloat16,
        vec![1_024],
    ));
    let error = PaddleOcrVl16Schema::validate(&unexpected).unwrap_err();
    assert_eq!(error.code(), SchemaErrorCode::UnexpectedTensor);
    assert_eq!(
        error.tensor_name(),
        Some("model.layers.18.input_layernorm.weight")
    );
}

#[test]
fn validator_detects_every_single_tensor_mutation_not_just_the_first_family() {
    let exact = observed_exact_catalog();
    for index in 0..exact.len() {
        let name = exact[index].name.clone();

        let mut missing = exact.clone();
        missing.remove(index);
        let error = PaddleOcrVl16Schema::validate(&missing).unwrap_err();
        assert_eq!(error.code(), SchemaErrorCode::MissingTensor, "{name}");
        assert_eq!(error.tensor_name(), Some(name.as_str()), "{name}");

        let mut wrong_shape = exact.clone();
        wrong_shape[index].shape[0] = wrong_shape[index].shape[0]
            .checked_add(1)
            .expect("fixture dimensions are small");
        let error = PaddleOcrVl16Schema::validate(&wrong_shape).unwrap_err();
        assert_eq!(error.code(), SchemaErrorCode::ShapeMismatch, "{name}");
        assert_eq!(error.tensor_name(), Some(name.as_str()), "{name}");

        let mut wrong_dtype = exact.clone();
        wrong_dtype[index].dtype = TensorDtype::Float16;
        let error = PaddleOcrVl16Schema::validate(&wrong_dtype).unwrap_err();
        assert_eq!(error.code(), SchemaErrorCode::DtypeMismatch, "{name}");
        assert_eq!(error.tensor_name(), Some(name.as_str()), "{name}");

        let mut duplicate = exact.clone();
        duplicate.push(exact[index].clone());
        let error = PaddleOcrVl16Schema::validate(&duplicate).unwrap_err();
        assert_eq!(error.code(), SchemaErrorCode::DuplicateTensor, "{name}");
        assert_eq!(error.tensor_name(), Some(name.as_str()), "{name}");
    }
}
