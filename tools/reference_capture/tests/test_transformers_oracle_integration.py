from __future__ import annotations

from pathlib import Path
import sys

import pytest
from blake3 import blake3
from transformers.cache_utils import DynamicCache

torch = pytest.importorskip("torch")

from pvlc_reference.model_lock import load_pinned_paddleocr_vl_16_lock
from pvlc_reference.trace_bundle import CaseSpec, TraceLevel
from pvlc_reference.transformers_oracle import TransformersOracle


REPO_ROOT = Path(__file__).parents[3]
REVISION = "66317acc4c9fc17bd154591ce650735cd2855f3e"
SNAPSHOT = REPO_ROOT / "models" / "snapshots" / REVISION
LOCK_PATH = REPO_ROOT / "models" / "paddleocr-vl-1.6.lock"
CASE_PATH = REPO_ROOT / "cases" / "smoke" / "cases" / "ocr-clean-latin.json"
IMAGE_PATH = REPO_ROOT / "cases" / "smoke" / "assets" / "ocr-clean-latin.png"


def _m6_live_deep_shapes() -> dict[str, tuple[int, ...]]:
    shapes = {
        f"decoder.layer.{layer_index:02d}.kv.{kind}": (1, 2, 332, 128)
        for layer_index in range(18)
        for kind in ("key", "value")
    }
    shapes.update(
        {
            "decoder.decode.00.attention_mask": (1, 333),
            "decoder.decode.00.cache_position": (1,),
            "decoder.decode.00.position_ids": (3, 1, 1),
            "decoder.decode.00.rope.cos": (3, 1, 1, 128),
            "decoder.decode.00.rope.sin": (3, 1, 1, 128),
            "decoder.decode.00.layer.00.input": (1, 1, 1024),
            "decoder.decode.00.layer.00.norm1": (1, 1, 1024),
            "decoder.decode.00.layer.00.q": (1, 1, 2048),
            "decoder.decode.00.layer.00.k": (1, 1, 256),
            "decoder.decode.00.layer.00.v": (1, 1, 256),
            "decoder.decode.00.layer.00.mrope.q": (1, 16, 1, 128),
            "decoder.decode.00.layer.00.mrope.k": (1, 2, 1, 128),
            "decoder.decode.00.layer.00.attention.context": (1, 1, 2048),
            "decoder.decode.00.layer.00.attention.output": (1, 1, 1024),
            "decoder.decode.00.layer.00.attention.residual": (1, 1, 1024),
            "decoder.decode.00.layer.00.norm2": (1, 1, 1024),
            "decoder.decode.00.layer.00.mlp.gate": (1, 1, 3072),
            "decoder.decode.00.layer.00.mlp.up": (1, 1, 3072),
            "decoder.decode.00.layer.00.mlp.activation": (1, 1, 3072),
            "decoder.decode.00.layer.00.mlp.down": (1, 1, 1024),
            "decoder.decode.00.final_norm": (1, 1, 1024),
            "decoder.decode.00.logits": (1, 1, 103424),
        }
    )
    shapes.update(
        {
            f"decoder.decode.00.layer.{layer_index:02d}.output": (1, 1, 1024)
            for layer_index in range(18)
        }
    )
    shapes.update(
        {
            f"decoder.decode.00.layer.{layer_index:02d}.kv.{kind}": (1, 2, 333, 128)
            for layer_index in range(18)
            for kind in ("key", "value")
        }
    )
    return shapes


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_local_multimodal_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    sections = mrope_section * 2
    cos = torch.cat(
        [chunk[index % 3] for index, chunk in enumerate(cos.split(sections, dim=-1))],
        dim=-1,
    ).unsqueeze(1)
    sin = torch.cat(
        [chunk[index % 3] for index, chunk in enumerate(sin.split(sections, dim=-1))],
        dim=-1,
    ).unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _hook_registry_snapshot(model: torch.nn.Module) -> tuple[tuple[object, ...], ...]:
    snapshot = []
    for module in model.modules():
        snapshot.append(
            (
                id(module),
                tuple((key, id(hook)) for key, hook in sorted(module._forward_hooks.items())),
                tuple((key, id(hook)) for key, hook in sorted(module._forward_pre_hooks.items())),
                tuple(
                    (key, id(hook))
                    for key, hook in sorted(getattr(module, "_backward_hooks", {}).items())
                ),
                tuple(
                    (key, id(hook))
                    for key, hook in sorted(getattr(module, "_backward_pre_hooks", {}).items())
                ),
            )
        )
    return tuple(snapshot)


def require_snapshot() -> None:
    if not SNAPSHOT.exists():
        pytest.skip("pinned model snapshot is not present")


@pytest.mark.oracle
def test_official_processor_matches_independently_captured_exact_contract() -> None:
    require_snapshot()
    lock = load_pinned_paddleocr_vl_16_lock(LOCK_PATH)
    case = CaseSpec.load(CASE_PATH)
    oracle = TransformersOracle(
        snapshot=SNAPSHOT,
        model_lock=lock,
        device="cpu",
        dtype="float32",
    )

    capture = oracle.capture_processor(case=case, image_path=IMAGE_PATH)

    expected_ids = (
        100273,
        2969,
        93963,
        93919,
        101305,
        *((100295,) * 319),
        101306,
        93972,
        2497,
        93963,
        23,
        92267,
        93963,
        23,
    )
    assert capture.input_ids == expected_ids
    assert capture.attention_mask == (1,) * 332
    assert capture.image_grid_thw == (1, 22, 58)
    assert capture.pixel_values_shape == (1276, 3, 14, 14)
    assert capture.placeholder_id == 100295
    assert capture.placeholder_count == 319
    assert capture.pixel_values_digest == (
        "blake3:5f1d2fe2573ea1d13f2751ca15299237f4d721142372650dc034ccb6015104d9"
    )
    assert capture.pixel_min == -1.0
    assert capture.pixel_max == 1.0
    assert capture.pixel_mean == pytest.approx(0.9470888376235962, abs=1e-9)
    assert capture.pixel_std == pytest.approx(0.2840694487094879, abs=1e-9)


@pytest.mark.oracle
def test_processor_repeat_capture_is_bit_identical() -> None:
    require_snapshot()
    lock = load_pinned_paddleocr_vl_16_lock(LOCK_PATH)
    case = CaseSpec.load(CASE_PATH)

    first = TransformersOracle(SNAPSHOT, lock, device="cpu", dtype="float32")
    second = TransformersOracle(SNAPSHOT, lock, device="cpu", dtype="float32")

    assert first.capture_processor(case, IMAGE_PATH).canonical_bytes() == second.capture_processor(
        case, IMAGE_PATH
    ).canonical_bytes()


@pytest.mark.oracle
@pytest.mark.oracle_slow
def test_manual_loop_matches_model_generate_on_real_mps_prefill_and_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_snapshot()
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available on this host")
    lock = load_pinned_paddleocr_vl_16_lock(LOCK_PATH)
    case = CaseSpec.load(CASE_PATH)
    oracle = TransformersOracle(
        snapshot=SNAPSHOT,
        model_lock=lock,
        device="mps",
        dtype="bfloat16",
    )
    model = oracle._load_model()
    vision_rope = model.visual.vision_model.encoder.rotary_pos_emb
    assert vision_rope.dim == 36
    assert vision_rope.theta == 10_000.0
    assert vision_rope.inv_freq.dtype == torch.float32
    assert tuple(vision_rope.inv_freq.shape) == (18,)
    assert torch.count_nonzero(vision_rope.inv_freq).item() == 0
    decoder_owner_module = sys.modules[type(model.model).__module__]
    baseline_decoder_rope_fn = decoder_owner_module.apply_multimodal_rotary_pos_emb

    def hook_count() -> int:
        return sum(
            len(module._forward_hooks)
            + len(module._forward_pre_hooks)
            + len(getattr(module, "_backward_hooks", {}))
            + len(getattr(module, "_backward_pre_hooks", {}))
            for module in model.modules()
        )

    baseline_hook_count = hook_count()
    baseline_hook_snapshot = _hook_registry_snapshot(model)
    decoder_abi_observations = []
    self_attn_abi_observations = []

    def _decoder_abi_probe(_module, _arguments, keyword_arguments):
        inputs_embeds = keyword_arguments.get("inputs_embeds")
        if not isinstance(inputs_embeds, torch.Tensor) or inputs_embeds.shape[1] != 1:
            return
        past_key_values = keyword_arguments.get("past_key_values")
        decoder_abi_observations.append(
            {
                "attention_mask": keyword_arguments["attention_mask"].detach().cpu().clone(),
                "cache_position_keyword_present": "cache_position" in keyword_arguments,
                "past_is_dynamic_cache": isinstance(past_key_values, DynamicCache),
                "past_length": (
                    int(past_key_values.get_seq_length())
                    if hasattr(past_key_values, "get_seq_length")
                    else -1
                ),
                "position_ids": keyword_arguments["position_ids"].detach().cpu().clone(),
                "use_cache": bool(keyword_arguments.get("use_cache", False)),
            }
        )

    def _self_attn_abi_probe(_module, arguments, keyword_arguments):
        hidden_states = (
            arguments[0]
            if arguments
            else keyword_arguments.get("hidden_states")
        )
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.shape[1] != 1:
            return
        attention_mask = keyword_arguments.get("attention_mask")
        self_attn_abi_observations.append(
            {
                "attention_mask": (
                    attention_mask.detach().cpu().clone()
                    if isinstance(attention_mask, torch.Tensor)
                    else attention_mask
                ),
                "cache_position": keyword_arguments["cache_position"].detach().cpu().clone(),
                "position_ids": keyword_arguments["position_ids"].detach().cpu().clone(),
            }
        )

    abi_handles = [
        model.model.register_forward_pre_hook(_decoder_abi_probe, with_kwargs=True),
        model.model.layers[0].self_attn.register_forward_pre_hook(
            _self_attn_abi_probe,
            with_kwargs=True,
        ),
    ]
    try:
        capture = oracle.capture_artifacts(
            case=case,
            image_path=IMAGE_PATH,
            max_new_tokens=2,
            trace_level=TraceLevel.L3,
        )
    finally:
        for handle in abi_handles:
            handle.remove()
    comparison = capture.comparison
    abi_sequence_length = int(
        capture.captured.processor_tensors["processor.input_ids"].shape[1]
    )
    probed_first_decode_decoder = [
        observation
        for observation in decoder_abi_observations
        if observation["use_cache"]
        and observation["past_length"] == abi_sequence_length
    ]
    assert probed_first_decode_decoder
    assert all(
        observation["past_is_dynamic_cache"]
        and not observation["cache_position_keyword_present"]
        and tuple(observation["attention_mask"].shape)
        == (1, abi_sequence_length + 1)
        and observation["attention_mask"].dtype == torch.int64
        for observation in probed_first_decode_decoder
    )
    probed_first_decode_self_attn = [
        observation
        for observation in self_attn_abi_observations
        if observation["cache_position"].tolist() == [abi_sequence_length]
    ]
    assert probed_first_decode_self_attn
    # The pinned Transformers causal-mask helper elides this all-visible mask.
    assert all(
        observation["attention_mask"] is None
        for observation in probed_first_decode_self_attn
    )

    assert comparison.generate_tokens == (94013, 898)
    assert comparison.manual_trace.tokens == comparison.generate_tokens
    assert comparison.decoded_text == "JUL"
    assert len(comparison.manual_trace.steps) == 2
    assert comparison.manual_trace.steps[0].chosen_token == 94013
    assert comparison.manual_trace.steps[1].chosen_token == 898
    assert all(step.top_tokens[0][0] == step.chosen_token for step in comparison.manual_trace.steps)
    assert set(capture.captured.processor_tensors) == {
        "processor.attention_mask",
        "processor.image_grid_thw",
        "processor.input_ids",
        "processor.pixel_values",
    }
    assert {
        name: tuple(tensor.shape)
        for name, tensor in capture.captured.processor_tensors.items()
    } == {
        "processor.attention_mask": (1, 332),
        "processor.image_grid_thw": (1, 3),
        "processor.input_ids": (1, 332),
        "processor.pixel_values": (1276, 3, 14, 14),
    }
    assert {
        name: tuple(tensor.shape)
        for name, tensor in capture.captured.stage_tensors.items()
    } == {
        "decoder.embedding": (1, 332, 1024),
        "decoder.mrope.delta": (1, 1),
        "decoder.mrope.index": (3, 1, 332),
        "decoder.prefill.logits.last": (1, 103424),
        "multimodal.image_token_indices": (319, 2),
        "multimodal.inputs_embeds": (1, 332, 1024),
        "projector.final": (319, 1024),
        "vision.final": (1276, 1152),
    }
    assert {
        name: tuple(tensor.shape)
        for name, tensor in capture.captured.deep_tensors.items()
    } == {
        "decoder.final_norm": (1, 332, 1024),
        "decoder.rope.cos": (3, 1, 332, 128),
        "decoder.rope.sin": (3, 1, 332, 128),
        "decoder.layer.00.input": (1, 332, 1024),
        "decoder.layer.00.norm1": (1, 332, 1024),
        "decoder.layer.00.q": (1, 332, 2048),
        "decoder.layer.00.k": (1, 332, 256),
        "decoder.layer.00.v": (1, 332, 256),
        "decoder.layer.00.mrope.q": (1, 16, 332, 128),
        "decoder.layer.00.mrope.k": (1, 2, 332, 128),
        "decoder.layer.00.kv.key": (1, 2, 332, 128),
        "decoder.layer.00.kv.value": (1, 2, 332, 128),
        "decoder.layer.00.attention.context": (1, 332, 2048),
        "decoder.layer.00.attention.output": (1, 332, 1024),
        "decoder.layer.00.attention.residual": (1, 332, 1024),
        "decoder.layer.00.norm2": (1, 332, 1024),
        "decoder.layer.00.mlp.gate": (1, 332, 3072),
        "decoder.layer.00.mlp.up": (1, 332, 3072),
        "decoder.layer.00.mlp.activation": (1, 332, 3072),
        "decoder.layer.00.mlp.down": (1, 332, 1024),
        **_m6_live_deep_shapes(),
        **{
            f"decoder.layer.{layer_index:02d}.output": (1, 332, 1024)
            for layer_index in range(18)
        },
        "projector.gelu": (319, 4608),
        "projector.linear1": (319, 4608),
        "projector.linear2": (319, 1024),
        "projector.merge": (319, 4608),
        "projector.pre_norm": (1276, 1152),
        "vision.embeddings.output": (1, 1276, 1152),
        "vision.embeddings.patch": (1, 1276, 1152),
        "vision.layer.00.attention.context": (1, 1276, 1152),
        "vision.layer.00.attention.output": (1, 1276, 1152),
        "vision.layer.00.attention.residual": (1, 1276, 1152),
        "vision.layer.00.k": (1, 1276, 1152),
        "vision.layer.00.mlp.activation": (1, 1276, 4304),
        "vision.layer.00.mlp.fc1": (1, 1276, 4304),
        "vision.layer.00.mlp.output": (1, 1276, 1152),
        "vision.layer.00.norm1": (1, 1276, 1152),
        "vision.layer.00.norm2": (1, 1276, 1152),
        "vision.layer.00.output": (1, 1276, 1152),
        "vision.layer.00.q": (1, 1276, 1152),
        "vision.layer.00.v": (1, 1276, 1152),
        "vision.layer.01.output": (1, 1276, 1152),
        "vision.layer.13.output": (1, 1276, 1152),
        "vision.layer.26.output": (1, 1276, 1152),
        "vision.rope.frequencies": (58, 18),
    }
    assert all(
        tensor.device.type == "cpu"
        for group in (
            capture.captured.processor_tensors,
            capture.captured.stage_tensors,
            capture.captured.deep_tensors,
        )
        for tensor in group.values()
    )
    assert capture.captured.token_trace == comparison.manual_trace
    assert decoder_owner_module.apply_multimodal_rotary_pos_emb is baseline_decoder_rope_fn
    assert _hook_registry_snapshot(model) == baseline_hook_snapshot
    processor_tensors = capture.captured.processor_tensors
    assert processor_tensors["processor.input_ids"].dtype == torch.int64
    assert processor_tensors["processor.attention_mask"].dtype == torch.int64
    assert processor_tensors["processor.image_grid_thw"].dtype == torch.int64
    assert processor_tensors["processor.pixel_values"].dtype == torch.float32
    assert tuple(processor_tensors["processor.input_ids"][0].tolist()) == (
        comparison.processor.input_ids
    )
    assert tuple(processor_tensors["processor.attention_mask"][0].tolist()) == (
        comparison.processor.attention_mask
    )
    assert tuple(processor_tensors["processor.image_grid_thw"][0].tolist()) == (
        comparison.processor.image_grid_thw
    )
    pixel_raw = (
        processor_tensors["processor.pixel_values"]
        .contiguous()
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    assert f"blake3:{blake3(pixel_raw).hexdigest()}" == (
        comparison.processor.pixel_values_digest
    )
    assert comparison.processor.pixel_values_digest == (
        "blake3:5f1d2fe2573ea1d13f2751ca15299237f4d721142372650dc034ccb6015104d9"
    )
    expected_value_anchors = {
        "decoder.embedding": (
            "blake3:a73b8cbb0ceff549fcba3cb6ef2e6f65bbf4b316bb0cc0f8ad58a94c25724d5a",
            -0.03369140625,
        ),
        "decoder.final_norm": (
            "blake3:2a69c20f24b2a517170611a29cb71fa46a0c3e8ee758805031d3fe9ea2318ac9",
            3.515625,
        ),
        "decoder.rope.cos": (
            "blake3:096287f2c2ee912105fbc747def39441b541c50b87b1330a8b3b3647b2b49654",
            1.0,
        ),
        "decoder.rope.sin": (
            "blake3:d34eff803104785331690d7f263c4f7ce44838f6083c5f2fb5ed987de613d310",
            0.0,
        ),
        "decoder.layer.00.input": (
            "blake3:8b46524fa1d413be6ee140b8af80c18547c3d505fd8f61c9781962e957f2da52",
            -0.03369140625,
        ),
        "decoder.layer.00.norm1": (
            "blake3:12ce0b7ba1b61c8edd12264be4178f9e566e78994cf64256b1c27f7cf8dcb76b",
            -0.90234375,
        ),
        "decoder.layer.00.q": (
            "blake3:888a4232edc8b6e404f34f494961b2e4645af3b6cebbd17f06ec66058b70b111",
            -0.5390625,
        ),
        "decoder.layer.00.k": (
            "blake3:e729075dff364dca4699edb9c1e9e96ea856cffc2c5e091d96899a642eb0c02a",
            0.052001953125,
        ),
        "decoder.layer.00.v": (
            "blake3:fc6f30bc2fc420c6166a0380c29c349c71caf1944dab6671aca50c5eb5f27202",
            -0.004119873046875,
        ),
        "decoder.layer.00.mrope.q": (
            "blake3:0562583ec6c4dd7aa401c26a45d3e8ae24e2088b84aa5d318b134f422a06766a",
            -0.5390625,
        ),
        "decoder.layer.00.mrope.k": (
            "blake3:a852c75f0d62dc96e5e8d6c81bc98b786c24eae2a3e6bde4e638c0bb60429c68",
            0.052001953125,
        ),
        "decoder.layer.00.kv.key": (
            "blake3:a852c75f0d62dc96e5e8d6c81bc98b786c24eae2a3e6bde4e638c0bb60429c68",
            0.052001953125,
        ),
        "decoder.layer.00.kv.value": (
            "blake3:a612337ce699b3c4577da81b8b292f954e2341726e870a16f0d55fb9fce3e7ed",
            -0.004119873046875,
        ),
        "decoder.layer.00.attention.context": (
            "blake3:8e7a9de666991e9320c6909e84f2f9b0fcb5e5f1dac5d192a4eb80a179077b01",
            -0.004119873046875,
        ),
        "decoder.layer.00.attention.output": (
            "blake3:4065a11aaa5fce98734cd37b6cb38291e01a826c735c6e3295db3fc29397aab8",
            -0.0001983642578125,
        ),
        "decoder.layer.00.attention.residual": (
            "blake3:fff4760d2fb144f525af4bde67ecee03a4f53fbfcfe6af6ff44ea4edb6f8eac3",
            -0.033935546875,
        ),
        "decoder.layer.00.norm2": (
            "blake3:497c88807b977ca91799ec9d1e8bc87f9b33df12b9299e02d69be8da2881d9bf",
            -1.078125,
        ),
        "decoder.layer.00.mlp.gate": (
            "blake3:6346f343d72cc55660073390ef3f84e138f7c7e46977806a38aba97fccafa24f",
            0.515625,
        ),
        "decoder.layer.00.mlp.up": (
            "blake3:339cfc0afe1fd98a2b78a45da5a0d89fc3bf99b77d0799d2bb13774a9ef6aeca",
            -0.34375,
        ),
        "decoder.layer.00.mlp.activation": (
            "blake3:f741f928b2e049f5063f455bfcdee515feaf6b43801aa065ea143863403bc1de",
            -0.11083984375,
        ),
        "decoder.layer.00.mlp.down": (
            "blake3:06d0dbd588511df0f72cf75092538941c19a9aedebbde59c986757182b5f3633",
            0.005767822265625,
        ),
        "decoder.layer.00.output": (
            "blake3:7130fabeb187b3b9dc463fa0aea6c39775a674ae6497513f0b48b0216c9cac6e",
            -0.0281982421875,
        ),
        "decoder.layer.01.output": (
            "blake3:616ae03b02d5cc638f6c3f8546529ae19b933fa454fbc4cd1ede5bca05b351cc",
            -2.0625,
        ),
        "decoder.layer.02.output": (
            "blake3:371450c422d098b0d449394fade6194edcdfbc61632be6e872377810dbfc8728",
            -2.046875,
        ),
        "decoder.layer.03.output": (
            "blake3:d025392f602ced5948a4892a78dc6a77e965ff78848333f1f6c37ae3c1ae2763",
            -2.03125,
        ),
        "decoder.layer.04.output": (
            "blake3:a35191b2b032d877eafdfacfe042154d919ba65a0814de29bff0ff1b7e9ad103",
            -2.046875,
        ),
        "decoder.layer.05.output": (
            "blake3:c9b136d5891238bdc9df6f2535c82179a92e7428dc5c13386d2b35ab81fc5655",
            -1.9296875,
        ),
        "decoder.layer.06.output": (
            "blake3:2e1e6dee728eb412b40e4dfc7c509ccf251ebaa90b14c3838b649558bbbfd3a4",
            -1.9609375,
        ),
        "decoder.layer.07.output": (
            "blake3:dafdf38e87de7f87b9f742b97153a51fed1854c5e40501c3af07155478747745",
            -1.984375,
        ),
        "decoder.layer.08.output": (
            "blake3:2d1172cb3ef44008b86ca171547963637695fc55655125f7557af5047319d52f",
            -2.046875,
        ),
        "decoder.layer.17.output": (
            "blake3:6e21cbbaa94f6e7dd979d8e039b59cdf86140569b3262d30489eaf6eb091ba20",
            31.875,
        ),
        "decoder.layer.09.output": (
            "blake3:51b0b2b6c7d57d0bcc3c062ef093aa902c4527da83acaafbc96098ffab184241",
            -1.984375,
        ),
        "decoder.layer.10.output": (
            "blake3:b1f36a6cedb0e4a52f2b8ba35e5906434e3b7575c638f910e380dd70d0c29eb1",
            -2.046875,
        ),
        "decoder.layer.11.output": (
            "blake3:9ea87539784ead4957837f8f2773fcaa327db36f7508f93744c3144dffbac9fc",
            -2.09375,
        ),
        "decoder.layer.12.output": (
            "blake3:cf8630fe39cb849e8e3756d389c332078aa2a197c30a912bfcb4eef6709c7b62",
            -2.109375,
        ),
        "decoder.layer.13.output": (
            "blake3:02bd1711796f8723c89cb7492556553af81f8f55fdda58c5ff891d1f226a8aa3",
            -2.125,
        ),
        "decoder.layer.14.output": (
            "blake3:82493b17983fc4c564dd8282fb8c88f1367b27826738df4380878ef85aa2c7ef",
            -2.171875,
        ),
        "decoder.layer.15.output": (
            "blake3:30e4e90ad96c8e90076bd9207bf2b3317d854cc8187c5c617116118527f61a28",
            -2.078125,
        ),
        "decoder.layer.16.output": (
            "blake3:450f9361fe27e0084145124767c1a294a9a33f4fd40460311a2ca58d9f1f814c",
            -1.78125,
        ),
        "decoder.mrope.delta": (
            "blake3:48378c7f3201de62c1fb0e040903669c4a925b3a9611aa4b1f712a8627d7787e",
            -290,
        ),
        "decoder.mrope.index": (
            "blake3:b51cce87812eafc3316d2606374eb1b9690db1286c9f418d7ca488b75d4c843b",
            0,
        ),
        "decoder.prefill.logits.last": (
            "blake3:d661fb880ccfcc073581609745c19bc512d37b67a79430f8072449a762288b8b",
            -3.3125,
        ),
        "multimodal.image_token_indices": (
            "blake3:3f439507ab0a107385cf9967f58d8242d74f735397eef4786185bb64d96adb4d",
            0,
        ),
        "multimodal.inputs_embeds": (
            "blake3:8b46524fa1d413be6ee140b8af80c18547c3d505fd8f61c9781962e957f2da52",
            -0.03369140625,
        ),
        "projector.final": (
            "blake3:494794a8a2f80db8b3b85ff005195ce83f50082a14dea97c6cdcae8faa94b658",
            1.5,
        ),
        "projector.gelu": (
            "blake3:4ca4dfd47be5f909a03dc4a3a1eddbf7f7f11dd3e1a965bcd1c48f75a51dd974",
            0.57421875,
        ),
        "projector.linear1": (
            "blake3:a591455215abbb29b52536fd040a605432adf3872342193f84adc77c809899e4",
            0.74609375,
        ),
        "projector.linear2": (
            "blake3:494794a8a2f80db8b3b85ff005195ce83f50082a14dea97c6cdcae8faa94b658",
            1.5,
        ),
        "projector.merge": (
            "blake3:5fdff15e62ca7c1d7610faf915e403a13d085153c4d9369bbe9e0b73251007fd",
            -0.09521484375,
        ),
        "projector.pre_norm": (
            "blake3:76749ee1301cabcd7909bf5b330c998295728676bc53b98205f71f2f4809b430",
            -0.09521484375,
        ),
        "vision.final": (
            "blake3:3f07cad9453c7702e06d2060f6685b3c47052b5d02ef6829790c43a27a18eb42",
            -0.2333984375,
        ),
        # The capture path enables deterministic algorithms before the first
        # official forward; MPS bilinear interpolation has a distinct stable
        # byte result in that mode.
        "vision.embeddings.output": (
            "blake3:d46a0d64f38f87888e4a1bb98e9f90f77612175d4a4d6254e69150f27201531a",
            1.125,
        ),
        "vision.embeddings.patch": (
            "blake3:0d2febe9efcf5a6825560e4d34f8511d16ed3ac45bf4d0536fed79e93183908e",
            1.171875,
        ),
        "vision.layer.00.attention.context": (
            "blake3:72a0862168aa7471380edcd5e1c98f2f39a5f6c5a317ed4bc0a5b3af802f3c7d",
            0.09228515625,
        ),
        "vision.layer.00.attention.output": (
            "blake3:9ad8db1231aeb7d00e13cbab4cfed1a304a0da221fec843924e24230e9a8db09",
            0.03564453125,
        ),
        "vision.layer.00.attention.residual": (
            "blake3:05fcb6d6e10ae5c7717be362f9618313ee5f78e02236149e581981052d51bdbc",
            1.1640625,
        ),
        "vision.layer.00.k": (
            "blake3:2f041b999dacd383cf497b7f627bf101e980397125e31c5801f360b5cb8a8836",
            -0.400390625,
        ),
        "vision.layer.00.mlp.activation": (
            "blake3:dad3d74c84bd56201da558bb4da8c9a78625769a0b9220b3aef35a7e5cb8683c",
            -0.0,
        ),
        "vision.layer.00.mlp.fc1": (
            "blake3:6aded502ffbb0dd30a37183d916d045a9115cc7e79639077983c65f6585b12fd",
            -5.8125,
        ),
        "vision.layer.00.mlp.output": (
            "blake3:c0750d4a37ff40db692c368b02fa8a937d0a1fea261bd634abc72504b81746d7",
            0.11865234375,
        ),
        "vision.layer.00.norm1": (
            "blake3:48f4e1590fda9d9e304386862ee999bff87dbafd682f1a544bd4298517448f0a",
            0.0002460479736328125,
        ),
        "vision.layer.00.norm2": (
            "blake3:7a9a88c2fa086c056c2b7a33aaa18bc6016b818773fb7b107021cb316746df2f",
            0.0081787109375,
        ),
        "vision.layer.00.output": (
            "blake3:6c766d8e3ff4905dad1618d0a6d7f5b0c34f346dcc02893a1cb282a3f69fb5b7",
            1.28125,
        ),
        "vision.layer.00.q": (
            "blake3:4b21fe817135fc3f9bf8d26b0e0cba98d3795c6e46db7f6ea2ab594569d049f6",
            -0.08349609375,
        ),
        "vision.layer.00.v": (
            "blake3:c8d1fd92ad9a9457c385c83e8901490400a333b3c7be84f6e67cee13c0e29da3",
            0.1005859375,
        ),
        "vision.layer.01.output": (
            "blake3:3f3fc21111fdfcc9b15286571717849b9cd7cefab5a891d05b699d07abf25b1c",
            1.109375,
        ),
        "vision.layer.13.output": (
            "blake3:5cbe9756a887540973d0b032be7bcda36e156ee81c55dee52c65d69d18540e8f",
            1.515625,
        ),
        "vision.layer.26.output": (
            "blake3:e746473f1549fb18081c8edb3d13d860792009b90a8959de1e579af5399467d9",
            -0.7109375,
        ),
        "vision.rope.frequencies": (
            "blake3:cde8b5be1bdb15331ce3ce22834c5914476e1ffd6371aa4e77e7bcf7e2086f5b",
            0.0,
        ),
    }
    all_checkpoints = {
        **capture.captured.stage_tensors,
        **capture.captured.deep_tensors,
    }
    unhashed_m6_ids = set(_m6_live_deep_shapes()) - set(expected_value_anchors)
    assert set(expected_value_anchors) == set(all_checkpoints) - unhashed_m6_ids
    assert unhashed_m6_ids <= set(all_checkpoints)
    integer_semantic_ids = {
        "decoder.mrope.delta",
        "decoder.mrope.index",
        "multimodal.image_token_indices",
    }
    for semantic_id, (expected_hash, expected_first) in expected_value_anchors.items():
        tensor = all_checkpoints[semantic_id]
        expected_dtype = (
            torch.int64
            if semantic_id in integer_semantic_ids
            else (
                torch.float32
                if semantic_id == "vision.rope.frequencies"
                else torch.bfloat16
            )
        )
        assert tensor.dtype == expected_dtype
        raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        assert f"blake3:{blake3(raw).hexdigest()}" == expected_hash
        assert float(tensor.flatten()[0]) == expected_first
    decode_metadata_ids = {
        "decoder.decode.00.attention_mask",
        "decoder.decode.00.cache_position",
        "decoder.decode.00.position_ids",
    }
    for semantic_id in unhashed_m6_ids:
        expected_dtype = torch.int64 if semantic_id in decode_metadata_ids else torch.bfloat16
        assert all_checkpoints[semantic_id].dtype == expected_dtype
    stage = capture.captured.stage_tensors
    image_indices = stage["multimodal.image_token_indices"]
    assert torch.equal(
        image_indices,
        torch.argwhere(processor_tensors["processor.input_ids"] == 100295),
    )
    expected_inputs_embeds = stage["decoder.embedding"].clone()
    expected_inputs_embeds[image_indices[:, 0], image_indices[:, 1]] = stage[
        "projector.final"
    ]
    assert torch.equal(expected_inputs_embeds, stage["multimodal.inputs_embeds"])

    rope_delta = int(stage["decoder.mrope.delta"][0, 0])
    assert all(step.rope_delta == rope_delta for step in comparison.manual_trace.steps)
    sequence_length = int(processor_tensors["processor.input_ids"].shape[1])
    first_decode_position = sequence_length + rope_delta
    assert first_decode_position == 42
    assert comparison.manual_trace.steps[0].position_ids == (
        first_decode_position,
        first_decode_position,
        first_decode_position,
    )

    mrope_index = stage["decoder.mrope.index"]
    active_mask = processor_tensors["processor.attention_mask"].to(torch.bool)
    assert tuple(mrope_index.shape[1:]) == tuple(active_mask.shape)
    active_positions = mrope_index[:, active_mask]
    assert tuple(active_positions.shape) == (3, int(active_mask.sum()))
    assert active_positions.shape[1] == sequence_length
    assert int(active_positions.min()) >= 0
    assert not torch.equal(
        capture.captured.deep_tensors["vision.embeddings.patch"],
        capture.captured.deep_tensors["vision.embeddings.output"],
    )
    deep = capture.captured.deep_tensors
    assert torch.equal(deep["decoder.layer.00.input"], stage["multimodal.inputs_embeds"])
    decoder_q = (
        deep["decoder.layer.00.q"].view(1, 332, 16, 128).transpose(1, 2).contiguous()
    )
    decoder_k = (
        deep["decoder.layer.00.k"].view(1, 332, 2, 128).transpose(1, 2).contiguous()
    )
    decoder_v = (
        deep["decoder.layer.00.v"].view(1, 332, 2, 128).transpose(1, 2).contiguous()
    )
    expected_mrope_q, expected_mrope_k = _apply_local_multimodal_rope(
        decoder_q,
        decoder_k,
        deep["decoder.rope.cos"],
        deep["decoder.rope.sin"],
        [16, 24, 24],
    )
    assert torch.equal(deep["decoder.layer.00.mrope.q"], expected_mrope_q)
    assert torch.equal(deep["decoder.layer.00.mrope.k"], expected_mrope_k)
    assert torch.equal(deep["decoder.layer.00.kv.key"], deep["decoder.layer.00.mrope.k"])
    assert torch.equal(deep["decoder.layer.00.kv.value"], decoder_v)

    decode_prefix = "decoder.decode.00"
    decode_attention_mask = deep[f"{decode_prefix}.attention_mask"]
    decode_cache_position = deep[f"{decode_prefix}.cache_position"]
    decode_position_ids = deep[f"{decode_prefix}.position_ids"]
    first_decode_decoder_observations = [
        observation
        for observation in decoder_abi_observations
        if observation["use_cache"] and observation["past_length"] == sequence_length
    ]
    assert first_decode_decoder_observations
    for observation in first_decode_decoder_observations:
        assert observation["past_is_dynamic_cache"]
        assert not observation["cache_position_keyword_present"]
        assert tuple(observation["attention_mask"].shape) == (1, sequence_length + 1)
        assert observation["attention_mask"].dtype == torch.int64
        assert tuple(observation["position_ids"].shape) == (3, 1, 1)
    assert any(
        torch.equal(observation["attention_mask"], decode_attention_mask)
        for observation in first_decode_decoder_observations
    )

    first_decode_self_attn_observations = [
        observation
        for observation in self_attn_abi_observations
        if observation["cache_position"].tolist() == [sequence_length]
    ]
    assert first_decode_self_attn_observations
    for observation in first_decode_self_attn_observations:
        assert observation["attention_mask"] is None
        assert tuple(observation["position_ids"].shape) == (3, 1, 1)
    assert any(
        torch.equal(observation["cache_position"], decode_cache_position)
        and torch.equal(observation["position_ids"], decode_position_ids)
        for observation in first_decode_self_attn_observations
    )
    assert decode_cache_position.tolist() == [sequence_length]
    assert torch.equal(
        decode_attention_mask[:, :-1],
        processor_tensors["processor.attention_mask"],
    )
    assert decode_attention_mask[:, -1:].tolist() == [[1]]
    assert decode_position_ids[:, 0, 0].tolist() == [42, 42, 42]
    assert tuple(decode_position_ids[:, 0, 0].tolist()) == (
        comparison.manual_trace.steps[0].position_ids
    )
    assert torch.equal(decode_position_ids, mrope_index[:, :, -1:] + 1)

    decode_q = (
        deep[f"{decode_prefix}.layer.00.q"]
        .view(1, 1, 16, 128)
        .transpose(1, 2)
        .contiguous()
    )
    decode_k = (
        deep[f"{decode_prefix}.layer.00.k"]
        .view(1, 1, 2, 128)
        .transpose(1, 2)
        .contiguous()
    )
    decode_v = (
        deep[f"{decode_prefix}.layer.00.v"]
        .view(1, 1, 2, 128)
        .transpose(1, 2)
        .contiguous()
    )
    expected_decode_mrope_q, expected_decode_mrope_k = _apply_local_multimodal_rope(
        decode_q,
        decode_k,
        deep[f"{decode_prefix}.rope.cos"],
        deep[f"{decode_prefix}.rope.sin"],
        [16, 24, 24],
    )
    assert torch.equal(
        deep[f"{decode_prefix}.layer.00.mrope.q"],
        expected_decode_mrope_q,
    )
    assert torch.equal(
        deep[f"{decode_prefix}.layer.00.mrope.k"],
        expected_decode_mrope_k,
    )

    for layer_index in range(18):
        prefill_key = deep[f"decoder.layer.{layer_index:02d}.kv.key"]
        prefill_value = deep[f"decoder.layer.{layer_index:02d}.kv.value"]
        decode_key = deep[f"{decode_prefix}.layer.{layer_index:02d}.kv.key"]
        decode_value = deep[f"{decode_prefix}.layer.{layer_index:02d}.kv.value"]
        assert decode_key.shape[2] == prefill_key.shape[2] + 1
        assert decode_value.shape[2] == prefill_value.shape[2] + 1
        assert torch.equal(decode_key[:, :, :-1, :], prefill_key)
        assert torch.equal(decode_value[:, :, :-1, :], prefill_value)
    assert torch.equal(
        deep[f"{decode_prefix}.layer.00.kv.key"][:, :, -1:, :],
        deep[f"{decode_prefix}.layer.00.mrope.k"],
    )
    assert torch.equal(
        deep[f"{decode_prefix}.layer.00.kv.value"][:, :, -1:, :],
        decode_v,
    )
    assert torch.equal(
        deep[f"{decode_prefix}.layer.00.attention.residual"],
        deep[f"{decode_prefix}.layer.00.input"]
        + deep[f"{decode_prefix}.layer.00.attention.output"],
    )
    expected_decode_activation = (
        torch.nn.functional.silu(deep[f"{decode_prefix}.layer.00.mlp.gate"].to("mps"))
        * deep[f"{decode_prefix}.layer.00.mlp.up"].to("mps")
    ).to("cpu")
    assert torch.equal(
        deep[f"{decode_prefix}.layer.00.mlp.activation"],
        expected_decode_activation,
    )
    assert torch.equal(
        deep[f"{decode_prefix}.layer.00.output"],
        deep[f"{decode_prefix}.layer.00.attention.residual"]
        + deep[f"{decode_prefix}.layer.00.mlp.down"],
    )
    assert int(torch.argmax(deep[f"{decode_prefix}.logits"][0, -1]).item()) == (
        comparison.manual_trace.steps[1].chosen_token
    )
    assert tuple(deep["decoder.layer.00.attention.context"].shape) == (1, 332, 2048)
    assert torch.equal(
        deep["decoder.layer.00.attention.residual"],
        deep["decoder.layer.00.input"] + deep["decoder.layer.00.attention.output"],
    )
    expected_decoder_activation = (
        torch.nn.functional.silu(deep["decoder.layer.00.mlp.gate"].to("mps"))
        * deep["decoder.layer.00.mlp.up"].to("mps")
    ).to("cpu")
    assert torch.equal(deep["decoder.layer.00.mlp.activation"], expected_decoder_activation)
    assert torch.equal(
        deep["decoder.layer.00.output"],
        deep["decoder.layer.00.attention.residual"] + deep["decoder.layer.00.mlp.down"],
    )
    for layer_index in range(17):
        current_layer = deep[f"decoder.layer.{layer_index:02d}.output"]
        next_layer = deep[f"decoder.layer.{layer_index + 1:02d}.output"]
        assert not torch.equal(current_layer, next_layer)
    expected_merge = (
        deep["projector.pre_norm"]
        .reshape(1, 11, 2, 29, 2, 1152)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(319, 4608)
    )
    assert torch.equal(deep["projector.merge"], expected_merge)
    assert torch.equal(
        deep["projector.linear2"], capture.captured.stage_tensors["projector.final"]
    )
    assert torch.count_nonzero(deep["vision.rope.frequencies"]).item() == 0
    assert torch.equal(
        deep["vision.layer.00.attention.residual"],
        deep["vision.embeddings.output"] + deep["vision.layer.00.attention.output"],
    )
    assert torch.equal(
        deep["vision.layer.00.output"],
        deep["vision.layer.00.attention.residual"]
        + deep["vision.layer.00.mlp.output"],
    )
    assert decoder_owner_module.apply_multimodal_rotary_pos_emb is baseline_decoder_rope_fn
    assert _hook_registry_snapshot(model) == baseline_hook_snapshot
    assert hook_count() == baseline_hook_count

    repeated = oracle.capture_artifacts(
        case=case,
        image_path=IMAGE_PATH,
        max_new_tokens=2,
        trace_level=TraceLevel.L3,
    )
    assert repeated.comparison == capture.comparison
    for group_name in ("processor_tensors", "stage_tensors", "deep_tensors"):
        first_group = getattr(capture.captured, group_name)
        repeated_group = getattr(repeated.captured, group_name)
        assert tuple(sorted(first_group)) == tuple(sorted(repeated_group))
        for semantic_id in first_group:
            first_raw = first_group[semantic_id].contiguous().view(torch.uint8).numpy().tobytes()
            repeated_raw = (
            repeated_group[semantic_id].contiguous().view(torch.uint8).numpy().tobytes()
            )
            assert first_raw == repeated_raw
    assert decoder_owner_module.apply_multimodal_rotary_pos_emb is baseline_decoder_rope_fn
    assert _hook_registry_snapshot(model) == baseline_hook_snapshot
    assert hook_count() == baseline_hook_count

    original_layer1_forward = model.model.layers[1].forward
    original_layer1_forward_self = original_layer1_forward.__self__
    original_layer1_forward_func = original_layer1_forward.__func__

    def fail_mid_forward(*args: object, **kwargs: object) -> object:
        raise RuntimeError("intentional mid-forward trace failure")

    with monkeypatch.context() as patch_context:
        patch_context.setattr(model.model.layers[1], "forward", fail_mid_forward)
        with pytest.raises(RuntimeError, match="intentional mid-forward trace failure"):
            oracle.capture_artifacts(
                case=case,
                image_path=IMAGE_PATH,
                max_new_tokens=2,
                trace_level=TraceLevel.L3,
            )
    restored_layer1_forward = model.model.layers[1].forward
    assert restored_layer1_forward.__self__ is original_layer1_forward_self
    assert restored_layer1_forward.__func__ is original_layer1_forward_func
    assert decoder_owner_module.apply_multimodal_rotary_pos_emb is baseline_decoder_rope_fn
    assert _hook_registry_snapshot(model) == baseline_hook_snapshot
    assert hook_count() == baseline_hook_count

    def fail_generate(*args: object, **kwargs: object) -> object:
        raise RuntimeError("intentional trace failure")

    monkeypatch.setattr(model, "generate", fail_generate)
    with pytest.raises(RuntimeError, match="intentional trace failure"):
        oracle.capture_artifacts(
            case=case,
            image_path=IMAGE_PATH,
            max_new_tokens=2,
            trace_level=TraceLevel.L3,
        )
    assert decoder_owner_module.apply_multimodal_rotary_pos_emb is baseline_decoder_rope_fn
    assert _hook_registry_snapshot(model) == baseline_hook_snapshot
    assert hook_count() == baseline_hook_count
