//! Exact physical and semantic schema for the single supported checkpoint.
//!
//! This crate deliberately contains no generic model discovery. A checkpoint is
//! accepted only when every physical tensor matches this catalog exactly.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::Serialize;

pub const MODEL_ID: &str = "PaddlePaddle/PaddleOCR-VL-1.6";
pub const MODEL_REVISION: &str = "66317acc4c9fc17bd154591ce650735cd2855f3e";
pub const COMPILER_MODEL_ABI: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum TensorDtype {
    Bool,
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Uint64,
    Int64,
    Float8E4M3,
    Float8E5M2,
    BFloat16,
    Float16,
    Float32,
    Float64,
}

impl TensorDtype {
    #[must_use]
    pub const fn safetensors_name(self) -> &'static str {
        match self {
            Self::Bool => "BOOL",
            Self::Uint8 => "U8",
            Self::Int8 => "I8",
            Self::Uint16 => "U16",
            Self::Int16 => "I16",
            Self::Uint32 => "U32",
            Self::Int32 => "I32",
            Self::Uint64 => "U64",
            Self::Int64 => "I64",
            Self::Float8E4M3 => "F8_E4M3",
            Self::Float8E5M2 => "F8_E5M2",
            Self::BFloat16 => "BF16",
            Self::Float16 => "F16",
            Self::Float32 => "F32",
            Self::Float64 => "F64",
        }
    }

    #[must_use]
    pub const fn byte_width(self) -> u64 {
        match self {
            Self::Bool | Self::Uint8 | Self::Int8 | Self::Float8E4M3 | Self::Float8E5M2 => 1,
            Self::Uint16 | Self::Int16 | Self::BFloat16 | Self::Float16 => 2,
            Self::Uint32 | Self::Int32 | Self::Float32 => 4,
            Self::Uint64 | Self::Int64 | Self::Float64 => 8,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorSpec {
    pub name: String,
    pub dtype: TensorDtype,
    pub shape: Vec<u64>,
    pub semantic_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservedTensor {
    pub name: String,
    pub dtype: TensorDtype,
    pub shape: Vec<u64>,
}

impl ObservedTensor {
    #[must_use]
    pub fn new(name: impl Into<String>, dtype: TensorDtype, shape: impl Into<Vec<u64>>) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape: shape.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SchemaErrorCode {
    DuplicateTensor,
    UnexpectedTensor,
    DtypeMismatch,
    ShapeMismatch,
    MissingTensor,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SchemaError {
    code: SchemaErrorCode,
    tensor_name: Option<String>,
}

impl SchemaError {
    #[must_use]
    pub const fn code(&self) -> SchemaErrorCode {
        self.code
    }

    #[must_use]
    pub fn tensor_name(&self) -> Option<&str> {
        self.tensor_name.as_deref()
    }

    fn tensor(code: SchemaErrorCode, name: impl Into<String>) -> Self {
        Self {
            code,
            tensor_name: Some(name.into()),
        }
    }
}

impl fmt::Display for SchemaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "checkpoint schema error {:?}", self.code)?;
        if let Some(name) = &self.tensor_name {
            write!(formatter, " for tensor {name:?}")?;
        }
        Ok(())
    }
}

impl Error for SchemaError {}

pub struct PaddleOcrVl16Schema;

impl PaddleOcrVl16Schema {
    /// Returns the complete catalog in bytewise tensor-name order.
    #[must_use]
    pub fn tensor_specs() -> Vec<TensorSpec> {
        build_tensor_specs()
    }

    /// Canonical JSONL-terminated catalog used as a reproducibility anchor.
    #[must_use]
    pub fn canonical_catalog_bytes() -> Vec<u8> {
        #[derive(Serialize)]
        struct Record<'a> {
            dtype: &'a str,
            name: &'a str,
            shape: &'a [u64],
        }

        let specs = Self::tensor_specs();
        let records: Vec<_> = specs
            .iter()
            .map(|spec| Record {
                dtype: spec.dtype.safetensors_name(),
                name: &spec.name,
                shape: &spec.shape,
            })
            .collect();
        canonical_json_line(&records)
    }

    /// Canonical physical-name to semantic-role map. This is kept separate from
    /// the physical catalog so layout/schema and trace identity can be audited
    /// independently.
    #[must_use]
    pub fn canonical_semantic_map_bytes() -> Vec<u8> {
        #[derive(Serialize)]
        struct Record<'a> {
            name: &'a str,
            semantic_id: &'a str,
        }

        let specs = Self::tensor_specs();
        let records: Vec<_> = specs
            .iter()
            .map(|spec| Record {
                name: &spec.name,
                semantic_id: &spec.semantic_id,
            })
            .collect();
        canonical_json_line(&records)
    }

    pub fn validate(observed: &[ObservedTensor]) -> Result<(), SchemaError> {
        let expected = Self::tensor_specs();
        let expected_by_name: BTreeMap<_, _> = expected
            .iter()
            .map(|spec| (spec.name.as_str(), spec))
            .collect();
        let mut seen = BTreeSet::new();

        for tensor in observed {
            if !seen.insert(tensor.name.as_str()) {
                return Err(SchemaError::tensor(
                    SchemaErrorCode::DuplicateTensor,
                    &tensor.name,
                ));
            }
            let Some(spec) = expected_by_name.get(tensor.name.as_str()) else {
                return Err(SchemaError::tensor(
                    SchemaErrorCode::UnexpectedTensor,
                    &tensor.name,
                ));
            };
            if tensor.dtype != spec.dtype {
                return Err(SchemaError::tensor(
                    SchemaErrorCode::DtypeMismatch,
                    &tensor.name,
                ));
            }
            if tensor.shape != spec.shape {
                return Err(SchemaError::tensor(
                    SchemaErrorCode::ShapeMismatch,
                    &tensor.name,
                ));
            }
        }

        for spec in &expected {
            if !seen.contains(spec.name.as_str()) {
                return Err(SchemaError::tensor(
                    SchemaErrorCode::MissingTensor,
                    &spec.name,
                ));
            }
        }
        Ok(())
    }
}

fn canonical_json_line<T: Serialize>(value: &T) -> Vec<u8> {
    let mut bytes = serde_json::to_vec(value).expect("fixed schema is always serializable");
    bytes.push(b'\n');
    bytes
}

fn add(
    specs: &mut Vec<TensorSpec>,
    name: impl Into<String>,
    shape: impl Into<Vec<u64>>,
    semantic_id: impl Into<String>,
) {
    specs.push(TensorSpec {
        name: name.into(),
        dtype: TensorDtype::BFloat16,
        shape: shape.into(),
        semantic_id: semantic_id.into(),
    });
}

fn build_tensor_specs() -> Vec<TensorSpec> {
    let mut out = Vec::with_capacity(620);

    add(
        &mut out,
        "lm_head.weight",
        [103_424, 1_024],
        "lm_head.weight",
    );
    for suffix in ["bias", "weight"] {
        let pre_norm_shape = [1_152];
        add(
            &mut out,
            format!("mlp_AR.pre_norm.{suffix}"),
            pre_norm_shape,
            format!("projector.pre_norm.{suffix}"),
        );
        let linear1_shape = if suffix == "bias" {
            vec![4_608]
        } else {
            vec![4_608, 4_608]
        };
        add(
            &mut out,
            format!("mlp_AR.linear_1.{suffix}"),
            linear1_shape,
            format!("projector.linear1.{suffix}"),
        );
        let linear2_shape = if suffix == "bias" {
            vec![1_024]
        } else {
            vec![1_024, 4_608]
        };
        add(
            &mut out,
            format!("mlp_AR.linear_2.{suffix}"),
            linear2_shape,
            format!("projector.linear2.{suffix}"),
        );
    }
    add(
        &mut out,
        "model.embed_tokens.weight",
        [103_424, 1_024],
        "decoder.embedding.weight",
    );
    add(
        &mut out,
        "model.norm.weight",
        [1_024],
        "decoder.final_norm.weight",
    );

    for layer in 0..18 {
        let physical = format!("model.layers.{layer}");
        let semantic = format!("decoder.layer.{layer:02}");
        add(
            &mut out,
            format!("{physical}.input_layernorm.weight"),
            [1_024],
            format!("{semantic}.norm1.weight"),
        );
        add(
            &mut out,
            format!("{physical}.post_attention_layernorm.weight"),
            [1_024],
            format!("{semantic}.norm2.weight"),
        );
        for (projection, semantic_projection, shape) in [
            ("q_proj", "q", [2_048, 1_024]),
            ("k_proj", "k", [256, 1_024]),
            ("v_proj", "v", [256, 1_024]),
            ("o_proj", "out", [1_024, 2_048]),
        ] {
            add(
                &mut out,
                format!("{physical}.self_attn.{projection}.weight"),
                shape,
                format!("{semantic}.attention.{semantic_projection}.weight"),
            );
        }
        for (projection, semantic_projection, shape) in [
            ("gate_proj", "gate", [3_072, 1_024]),
            ("up_proj", "up", [3_072, 1_024]),
            ("down_proj", "down", [1_024, 3_072]),
        ] {
            add(
                &mut out,
                format!("{physical}.mlp.{projection}.weight"),
                shape,
                format!("{semantic}.mlp.{semantic_projection}.weight"),
            );
        }
    }

    add(
        &mut out,
        "visual.vision_model.embeddings.packing_position_embedding.weight",
        [32_768, 1_152],
        "vision.embeddings.packing_position.weight",
    );
    add(
        &mut out,
        "visual.vision_model.embeddings.patch_embedding.bias",
        [1_152],
        "vision.embeddings.patch.bias",
    );
    add(
        &mut out,
        "visual.vision_model.embeddings.patch_embedding.weight",
        [1_152, 3, 14, 14],
        "vision.embeddings.patch.weight",
    );
    add(
        &mut out,
        "visual.vision_model.embeddings.position_embedding.weight",
        [729, 1_152],
        "vision.embeddings.position.weight",
    );

    for layer in 0..27 {
        let physical = format!("visual.vision_model.encoder.layers.{layer}");
        let semantic = format!("vision.layer.{layer:02}");
        for (physical_norm, semantic_norm) in [("layer_norm1", "norm1"), ("layer_norm2", "norm2")] {
            for suffix in ["bias", "weight"] {
                add(
                    &mut out,
                    format!("{physical}.{physical_norm}.{suffix}"),
                    [1_152],
                    format!("{semantic}.{semantic_norm}.{suffix}"),
                );
            }
        }
        for (projection, bias_shape, weight_shape) in [
            ("fc1", vec![4_304], vec![4_304, 1_152]),
            ("fc2", vec![1_152], vec![1_152, 4_304]),
        ] {
            add(
                &mut out,
                format!("{physical}.mlp.{projection}.bias"),
                bias_shape,
                format!("{semantic}.mlp.{projection}.bias"),
            );
            add(
                &mut out,
                format!("{physical}.mlp.{projection}.weight"),
                weight_shape,
                format!("{semantic}.mlp.{projection}.weight"),
            );
        }
        for (physical_projection, semantic_projection) in [
            ("q_proj", "q"),
            ("k_proj", "k"),
            ("v_proj", "v"),
            ("out_proj", "out"),
        ] {
            for suffix in ["bias", "weight"] {
                let shape = if suffix == "bias" {
                    vec![1_152]
                } else {
                    vec![1_152, 1_152]
                };
                add(
                    &mut out,
                    format!("{physical}.self_attn.{physical_projection}.{suffix}"),
                    shape,
                    format!("{semantic}.attention.{semantic_projection}.{suffix}"),
                );
            }
        }
    }

    for suffix in ["bias", "weight"] {
        let in_shape = if suffix == "bias" {
            vec![3_456]
        } else {
            vec![3_456, 1_152]
        };
        add(
            &mut out,
            format!("visual.vision_model.head.attention.in_proj_{suffix}"),
            in_shape,
            format!("vision.head.attention.qkv.{suffix}"),
        );
        let out_shape = if suffix == "bias" {
            vec![1_152]
        } else {
            vec![1_152, 1_152]
        };
        add(
            &mut out,
            format!("visual.vision_model.head.attention.out_proj.{suffix}"),
            out_shape,
            format!("vision.head.attention.out.{suffix}"),
        );
        add(
            &mut out,
            format!("visual.vision_model.head.layernorm.{suffix}"),
            [1_152],
            format!("vision.head.norm.{suffix}"),
        );
        for (projection, bias_shape, weight_shape) in [
            ("fc1", vec![4_304], vec![4_304, 1_152]),
            ("fc2", vec![1_152], vec![1_152, 4_304]),
        ] {
            let shape = if suffix == "bias" {
                bias_shape
            } else {
                weight_shape
            };
            add(
                &mut out,
                format!("visual.vision_model.head.mlp.{projection}.{suffix}"),
                shape,
                format!("vision.head.mlp.{projection}.{suffix}"),
            );
        }
        add(
            &mut out,
            format!("visual.vision_model.post_layernorm.{suffix}"),
            [1_152],
            format!("vision.post_norm.{suffix}"),
        );
    }
    add(
        &mut out,
        "visual.vision_model.head.probe",
        [1, 1, 1_152],
        "vision.head.probe",
    );

    out.sort_unstable_by(|left, right| left.name.cmp(&right.name));
    debug_assert_eq!(out.len(), 620);
    debug_assert!(out.windows(2).all(|window| window[0].name < window[1].name));
    out
}
