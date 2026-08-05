from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
import errno
from functools import lru_cache
import importlib
import json
import os
from pathlib import Path
import signal
import shutil
import stat
import subprocess
import sys
import threading

import pytest
import torch
import torch.nn.functional as F
from blake3 import blake3
from safetensors import safe_open
from safetensors.torch import load_file

from pvlc_reference.capture import GenerationStep, GenerationTrace
from pvlc_reference.capture_artifacts import (
    CapturedArtifacts,
    export_golden_bundle,
    serialize_safetensors,
)
from pvlc_reference.golden_lock import GoldenEntry, GoldenLock
from pvlc_reference.trace_bundle import (
    CaptureProvenance,
    CaseSpec,
    TraceLevel,
    canonical_json_bytes,
    verify_bundle,
)


S = 332
L = 18
KVH = 2
D = 128
H = 1024
I = 3072
V = 103424
QH = 16

PREFILL_TOKEN = 94013
DECODE_TOKEN = 898
REPO_ROOT = Path(__file__).parents[3]
SMOKE_CASE_PATH = REPO_ROOT / "cases" / "smoke" / "cases" / "ocr-clean-latin.json"
SMOKE_IMAGE_PATH = REPO_ROOT / "cases" / "smoke" / "assets" / "ocr-clean-latin.png"
OFFICIAL_BUNDLE_DIGEST = (
    "blake3:47d07a39b638bc6ff68f1eaeab1a81af0407ab26338c745942cfd7e5c5faaa99"
)
OFFICIAL_SEMANTIC_FINGERPRINT = (
    "blake3:d5c7ec3c2be5cc1c3d6ed416f2ae8659ab3e5b2e851cee35afe7df10f436a9bc"
)
OFFICIAL_ARTIFACT_PATH = "artifacts/goldens/ocr.clean_latin.0001-l3"
M5_LAYER0_FIXTURE_PATH = (
    REPO_ROOT
    / "crates"
    / "pvlc-testkit"
    / "tests"
    / "fixtures"
    / "decoder-layer0-official-v1.safetensors"
)
POST_VERIFY_AUTHENTICATED_ARTIFACTS = (
    "hashes.json",
    "manifest.json",
    "processor.safetensors",
    "stage-checkpoints.safetensors",
    "deep-checkpoints.safetensors",
    "token-trace.jsonl",
)


def synthetic_capture(seed: int) -> tuple[CapturedArtifacts, CaptureProvenance]:
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def bf16_randn(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(
            torch.bfloat16
        )

    def rms_norm(tensor: torch.Tensor) -> torch.Tensor:
        source = tensor.to(torch.float32)
        return (source * torch.rsqrt(source.square().mean(dim=-1, keepdim=True) + 1e-5)).to(
            torch.bfloat16
        )

    input_ids = torch.randint(0, V, (1, S), generator=generator, dtype=torch.int64)
    input_ids[0, -1] = 23
    processor_tensors = {
        "processor.attention_mask": torch.ones((1, S), dtype=torch.int64),
        "processor.image_grid_thw": torch.tensor([[1, 22, 58]], dtype=torch.int64),
        "processor.input_ids": input_ids,
        "processor.pixel_values": torch.zeros(
            (22 * 58, 3, 14, 14), dtype=torch.float32
        ),
    }

    prefill_logits = torch.zeros((1, V), dtype=torch.bfloat16)
    prefill_logits[0, PREFILL_TOKEN] = 8.75
    prefill_logits[0, 93992] = 7.5
    stage_tensors = {
        "decoder.embedding": bf16_randn(1, S, H),
        "decoder.mrope.delta": torch.tensor([[-290]], dtype=torch.int64),
        "decoder.mrope.index": torch.arange(S, dtype=torch.int64)
        .clamp_max(41)
        .view(1, 1, S)
        .expand(3, 1, S)
        .clone(),
        "decoder.prefill.logits.last": prefill_logits,
    }

    prefix = "decoder.decode.00"
    deep_tensors: dict[str, torch.Tensor] = {
        f"{prefix}.attention_mask": torch.cat(
            (processor_tensors["processor.attention_mask"], torch.ones((1, 1), dtype=torch.int64)),
            dim=1,
        ),
        f"{prefix}.cache_position": torch.tensor([S], dtype=torch.int64),
        f"{prefix}.position_ids": torch.full((3, 1, 1), 42, dtype=torch.int64),
        f"{prefix}.rope.cos": bf16_randn(3, 1, 1, D),
        f"{prefix}.rope.sin": bf16_randn(3, 1, 1, D),
    }

    layer0_input = bf16_randn(1, 1, H)
    layer0_q = bf16_randn(1, 1, QH * D)
    layer0_k = bf16_randn(1, 1, KVH * D)
    layer0_v = bf16_randn(1, 1, KVH * D)
    layer0_mrope_q = bf16_randn(1, QH, 1, D)
    layer0_mrope_k = bf16_randn(1, KVH, 1, D)
    layer0_context = bf16_randn(1, 1, QH * D)
    layer0_attention_output = bf16_randn(1, 1, H)
    layer0_residual = (layer0_input + layer0_attention_output).to(torch.bfloat16)
    layer0_gate = bf16_randn(1, 1, I)
    layer0_up = bf16_randn(1, 1, I)
    layer0_activation = (
        F.silu(layer0_gate.to(torch.float32)) * layer0_up.to(torch.float32)
    ).to(torch.bfloat16)
    layer0_down = bf16_randn(1, 1, H)
    layer0_output = (layer0_residual + layer0_down).to(torch.bfloat16)
    deep_tensors.update(
        {
            f"{prefix}.layer.00.input": layer0_input,
            f"{prefix}.layer.00.norm1": rms_norm(layer0_input),
            f"{prefix}.layer.00.q": layer0_q,
            f"{prefix}.layer.00.k": layer0_k,
            f"{prefix}.layer.00.v": layer0_v,
            f"{prefix}.layer.00.mrope.q": layer0_mrope_q,
            f"{prefix}.layer.00.mrope.k": layer0_mrope_k,
            f"{prefix}.layer.00.attention.context": layer0_context,
            f"{prefix}.layer.00.attention.output": layer0_attention_output,
            f"{prefix}.layer.00.attention.residual": layer0_residual,
            f"{prefix}.layer.00.norm2": rms_norm(layer0_residual),
            f"{prefix}.layer.00.mlp.gate": layer0_gate,
            f"{prefix}.layer.00.mlp.up": layer0_up,
            f"{prefix}.layer.00.mlp.activation": layer0_activation,
            f"{prefix}.layer.00.mlp.down": layer0_down,
        }
    )

    current_output = layer0_output
    for layer_index in range(L):
        layer = f"{layer_index:02d}"
        prefill_key = bf16_randn(1, KVH, S, D)
        prefill_value = bf16_randn(1, KVH, S, D)
        appended_key = (
            layer0_mrope_k if layer_index == 0 else bf16_randn(1, KVH, 1, D)
        )
        appended_value = (
            layer0_v.view(1, 1, KVH, D).transpose(1, 2).contiguous()
            if layer_index == 0
            else bf16_randn(1, KVH, 1, D)
        )
        deep_tensors[f"decoder.layer.{layer}.kv.key"] = prefill_key
        deep_tensors[f"decoder.layer.{layer}.kv.value"] = prefill_value
        deep_tensors[f"{prefix}.layer.{layer}.kv.key"] = torch.cat(
            (prefill_key, appended_key), dim=2
        )
        deep_tensors[f"{prefix}.layer.{layer}.kv.value"] = torch.cat(
            (prefill_value, appended_value), dim=2
        )

        if layer_index:
            current_output = (
                current_output + bf16_randn(1, 1, H) * 0.03125
            ).to(torch.bfloat16)
        deep_tensors[f"{prefix}.layer.{layer}.output"] = current_output

    deep_tensors[f"{prefix}.final_norm"] = rms_norm(current_output)
    decode_logits = torch.zeros((1, 1, V), dtype=torch.bfloat16)
    decode_logits[0, 0, DECODE_TOKEN] = 8.5
    decode_logits[0, 0, 820] = 8.1875
    deep_tensors[f"{prefix}.logits"] = decode_logits

    prefill_kv_shapes = ((1, KVH, S, D),) * L
    post_kv_shapes = ((1, KVH, S + 1, D),) * L
    trace = GenerationTrace(
        tokens=(PREFILL_TOKEN, DECODE_TOKEN),
        steps=(
            GenerationStep(
                step=0,
                input_token=23,
                position_ids=(42, 42, 42),
                cache_position=S,
                rope_delta=-290,
                top_tokens=((PREFILL_TOKEN, 8.75), (93992, 7.5)),
                chosen_token=PREFILL_TOKEN,
                kv_shapes=prefill_kv_shapes,
            ),
            GenerationStep(
                step=1,
                input_token=PREFILL_TOKEN,
                position_ids=(43, 43, 43),
                cache_position=S + 1,
                rope_delta=-290,
                top_tokens=((DECODE_TOKEN, 8.5), (820, 8.1875)),
                chosen_token=DECODE_TOKEN,
                kv_shapes=post_kv_shapes,
            ),
        ),
    )
    provenance = CaptureProvenance(
        model_id="PaddlePaddle/PaddleOCR-VL-1.6",
        model_revision="66317acc4c9fc17bd154591ce650735cd2855f3e",
        model_lock_hash=(
            "blake3:c2ffa46821fdda1a7e4cd6bad63cdcdc2f775b8fed250b0679cc7d5411577d10"
        ),
        trace_schema_version=1,
        capture_tool_version="0.1.0",
        compatibility_shims=("paddleocr-vl-1.6/transformers-v5-abi@1",),
        python_version="3.12.13",
        torch_version="2.13.0",
        transformers_version="5.14.1",
        device="mps",
        dtype="bfloat16",
        deterministic_algorithms=True,
    )
    return (
        CapturedArtifacts(
            processor_tensors=processor_tensors,
            stage_tensors=stage_tensors,
            deep_tensors=deep_tensors,
            token_trace=trace,
        ),
        provenance,
    )


def _make_source(
    api,
    seed: int,
    *,
    captured: CapturedArtifacts | None = None,
    provenance: CaptureProvenance | None = None,
    publication: GoldenEntry | None = None,
    publication_lock_blake3: str | None = None,
):
    if captured is None or provenance is None:
        default_capture, default_provenance = synthetic_capture(seed)
    else:
        default_capture, default_provenance = captured, provenance
    selected_capture = default_capture if captured is None else captured
    selected_provenance = default_provenance if provenance is None else provenance
    selected_publication = publication or GoldenEntry(
        case_id="ocr.clean_latin.0001",
        trace_level=TraceLevel.L3,
        artifact_path=OFFICIAL_ARTIFACT_PATH,
        bundle_digest=OFFICIAL_BUNDLE_DIGEST,
        semantic_fingerprint=OFFICIAL_SEMANTIC_FINGERPRINT,
        generated_tokens=(PREFILL_TOKEN, DECODE_TOKEN),
        decoded_text="JUL",
        repeat_count=2,
    )
    return api.DecodeFixtureSource(
        captured=selected_capture,
        provenance=selected_provenance,
        publication=selected_publication,
        publication_lock_blake3=(
            _publication_lock_blake3(selected_publication)
            if publication_lock_blake3 is None
            else publication_lock_blake3
        ),
    )


def _publication_lock(publication: GoldenEntry) -> GoldenLock:
    return GoldenLock(
        format_version=1,
        model_revision="66317acc4c9fc17bd154591ce650735cd2855f3e",
        trace_schema_version=1,
        bundles=(publication,),
    )


def _publication_lock_blake3(publication: GoldenEntry) -> str:
    payload = _publication_lock(publication).canonical_bytes()
    return f"blake3:{blake3(payload).hexdigest()}"


def _replace_publication(source, **changes):
    publication = replace(source.publication, **changes)
    return replace(
        source,
        publication=publication,
        publication_lock_blake3=_publication_lock_blake3(publication),
    )


def _expected_output_tensors(source) -> dict[str, torch.Tensor]:
    captured = source.captured
    trace = captured.token_trace
    assert trace is not None
    prefix = "decoder.decode.00"
    deep = captured.deep_tensors
    expected: dict[str, torch.Tensor] = {}
    for kind in ("key", "value"):
        layers: list[torch.Tensor] = []
        for layer_index in range(L):
            layer = f"{layer_index:02d}"
            prefill = deep[f"decoder.layer.{layer}.kv.{kind}"]
            post = deep[f"{prefix}.layer.{layer}.kv.{kind}"]
            layers.append(
                torch.cat(
                    (
                        prefill[0].permute(1, 0, 2).contiguous(),
                        post[0, :, S, :].unsqueeze(0),
                    ),
                    dim=0,
                )
            )
        expected[f"{prefix}.kv.{kind}.layer_token_major"] = torch.stack(layers)

    expected.update(
        {
            f"{prefix}.attention_mask": deep[f"{prefix}.attention_mask"],
            f"{prefix}.cache_position": deep[f"{prefix}.cache_position"],
            f"{prefix}.input_token_id": torch.tensor(
                [[trace.steps[1].input_token]], dtype=torch.int64
            ),
            f"{prefix}.position_ids": deep[f"{prefix}.position_ids"],
            "decoder.mrope.delta": captured.stage_tensors["decoder.mrope.delta"],
            f"{prefix}.rope.cos.axis_major": deep[f"{prefix}.rope.cos"].squeeze(1),
            f"{prefix}.rope.sin.axis_major": deep[f"{prefix}.rope.sin"].squeeze(1),
        }
    )
    for suffix in (
        "input",
        "norm1",
        "q",
        "k",
        "v",
        "attention.output",
        "attention.residual",
        "norm2",
        "mlp.gate",
        "mlp.up",
        "mlp.activation",
        "mlp.down",
    ):
        semantic_id = f"{prefix}.layer.00.{suffix}"
        expected[semantic_id] = deep[semantic_id].squeeze(0)
    for suffix in ("mrope.q", "mrope.k"):
        source_id = f"{prefix}.layer.00.{suffix}"
        expected[f"{source_id}.token_major"] = (
            deep[source_id].squeeze(0).permute(1, 0, 2).contiguous()
        )
    context_id = f"{prefix}.layer.00.attention.context"
    expected[f"{context_id}.token_major"] = deep[context_id].reshape(1, QH, D)
    for layer_index in range(L):
        output_id = f"{prefix}.layer.{layer_index:02d}.output"
        expected[output_id] = deep[output_id].squeeze(0)
    expected[f"{prefix}.final_norm"] = deep[f"{prefix}.final_norm"].squeeze(0)
    expected[f"{prefix}.logits"] = deep[f"{prefix}.logits"].squeeze(0)
    assert len(expected) == 44
    return expected


def _replace_capture_tensor(
    captured: CapturedArtifacts,
    mapping_name: str,
    semantic_id: str,
    tensor: torch.Tensor,
) -> CapturedArtifacts:
    mapping = dict(getattr(captured, mapping_name))
    mapping[semantic_id] = tensor
    return replace(captured, **{mapping_name: mapping})


def _snapshot_source_tensors(
    *captures: CapturedArtifacts,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
    seen: set[int] = set()
    for captured in captures:
        for mapping in (
            captured.processor_tensors,
            captured.stage_tensors,
            captured.deep_tensors,
        ):
            for tensor in mapping.values():
                identity = id(tensor)
                if identity not in seen:
                    seen.add(identity)
                    snapshots.append((tensor, tensor.clone()))
    return tuple(snapshots)


def _stat_identity(metadata: os.stat_result) -> tuple[int, int]:
    return (metadata.st_dev, metadata.st_ino)


def _open_directory(path: Path) -> int:
    return os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )


def _entry_exists(name: str, *, directory_fd: int) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _read_bytes_at(path: str | os.PathLike[str], *, directory_fd: int | None) -> bytes:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        dir_fd=directory_fd,
    )
    try:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def _write_bytes_exclusive_at(
    name: str,
    payload: bytes,
    *,
    directory_fd: int,
) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
        dir_fd=directory_fd,
    )
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            assert written > 0
            remaining = remaining[written:]
    finally:
        os.close(descriptor)


def _write_descriptor_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        assert written > 0
        remaining = remaining[written:]


def _read_descriptor_bytes(descriptor: int) -> bytes:
    position = os.lseek(descriptor, 0, os.SEEK_CUR)
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.lseek(descriptor, position, os.SEEK_SET)


def _matching_entry_names(
    directory_fd: int,
    metadata: os.stat_result,
) -> list[str]:
    identity = _stat_identity(metadata)
    return sorted(
        name
        for name in os.listdir(directory_fd)
        if _stat_identity(
            os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        )
        == identity
    )


def _entry_names_for_descriptor(directory_fd: int, descriptor: int) -> list[str]:
    return _matching_entry_names(directory_fd, os.fstat(descriptor))


@dataclass(frozen=True)
class _CommitObservation:
    mode: str
    parent_fd: int
    output_name: str
    staging_fd: int | None = None
    source_name: str | None = None


def _read_commit_source_bytes(observation: _CommitObservation) -> bytes:
    if observation.mode == "fd":
        assert observation.staging_fd is not None
        return _read_descriptor_bytes(observation.staging_fd)
    assert observation.source_name is not None
    return _read_bytes_at(observation.source_name, directory_fd=observation.parent_fd)


def _commit_source_entry_names(observation: _CommitObservation) -> list[str]:
    if observation.mode == "fd":
        assert observation.staging_fd is not None
        return _entry_names_for_descriptor(observation.parent_fd, observation.staging_fd)
    assert observation.source_name is not None
    entry_stat = os.stat(
        observation.source_name,
        dir_fd=observation.parent_fd,
        follow_symlinks=False,
    )
    return _matching_entry_names(observation.parent_fd, entry_stat)


def _install_commit_wrapper(
    api,
    monkeypatch: pytest.MonkeyPatch,
    *,
    output_parent: Path,
    output_name: str,
    parent_identity: tuple[int, int] | None,
    handler,
) -> str:
    def authenticated_parent_identity(parent_fd: int) -> tuple[int, int]:
        descriptor_identity = _stat_identity(os.fstat(parent_fd))
        if parent_identity is not None:
            assert descriptor_identity == parent_identity
        lexical_stat = os.stat(output_parent, follow_symlinks=False)
        assert stat.S_ISDIR(lexical_stat.st_mode)
        assert _stat_identity(lexical_stat) == descriptor_identity
        return descriptor_identity

    commit_descriptor = getattr(api, "_commit_staging_descriptor", None)
    if callable(commit_descriptor):
        original_commit = commit_descriptor
        expected_output_name = output_name

        def wrapped_commit(staging_fd: int, *, parent_fd: int, output_name: str):
            authenticated_parent_identity(parent_fd)
            assert output_name == expected_output_name
            observation = _CommitObservation(
                mode="fd",
                parent_fd=parent_fd,
                output_name=output_name,
                staging_fd=staging_fd,
            )
            return handler(
                observation,
                lambda: original_commit(
                    staging_fd,
                    parent_fd=parent_fd,
                    output_name=output_name,
                ),
            )

        monkeypatch.setattr(api, "_commit_staging_descriptor", wrapped_commit)
        return "fd"

    original_link = api.os.link

    def wrapped_link(source_path, destination_path, *args, **kwargs):
        parent_fd = kwargs.get("src_dir_fd")
        assert isinstance(parent_fd, int)
        observed_parent_identity = authenticated_parent_identity(parent_fd)
        _inspect_publication_link(
            source_path,
            destination_path,
            kwargs,
            output_parent=output_parent,
            output_name=output_name,
            parent_identity=observed_parent_identity,
        )
        observation = _CommitObservation(
            mode="link",
            parent_fd=parent_fd,
            output_name=output_name,
            source_name=Path(source_path).name,
        )
        return handler(
            observation,
            lambda: original_link(source_path, destination_path, *args, **kwargs),
        )

    monkeypatch.setattr(api.os, "link", wrapped_link)
    return "link"


def _inspect_publication_link(
    source_path,
    destination_path,
    kwargs: dict[str, object],
    *,
    output_parent: Path,
    output_name: str,
    parent_identity: tuple[int, int],
) -> tuple[str, str, bool]:
    source = Path(source_path)
    destination = Path(destination_path)
    source_directory_fd = kwargs.get("src_dir_fd")
    destination_directory_fd = kwargs.get("dst_dir_fd")
    source_is_relative = not source.is_absolute()
    destination_is_relative = not destination.is_absolute()
    assert source_is_relative == destination_is_relative

    if source_is_relative:
        assert source == Path(source.name)
        assert destination == Path(destination.name)
        assert isinstance(source_directory_fd, int)
        assert isinstance(destination_directory_fd, int)
        assert _stat_identity(os.fstat(source_directory_fd)) == parent_identity
        assert _stat_identity(os.fstat(destination_directory_fd)) == parent_identity
        assert kwargs.get("follow_symlinks") is False
        assert destination.name == output_name
    else:
        assert source_directory_fd is None
        assert destination_directory_fd is None
        assert source.parent == output_parent
        assert destination == output_parent / output_name

    assert source.name.startswith(f".{output_name}.tmp-")
    return (source.name, destination.name, source_is_relative)


def _read_link_source_bytes(source_path, kwargs: dict[str, object]) -> bytes:
    source_directory_fd = kwargs.get("src_dir_fd")
    assert source_directory_fd is None or isinstance(source_directory_fd, int)
    return _read_bytes_at(source_path, directory_fd=source_directory_fd)


def _corrupt_source(source, case: str):
    captured = source.captured
    prefix = "decoder.decode.00"

    if case == "missing_tensor":
        deep = dict(captured.deep_tensors)
        del deep[f"{prefix}.final_norm"]
        return replace(source, captured=replace(captured, deep_tensors=deep))

    cache_id = f"{prefix}.layer.03.kv.key"
    cache = captured.deep_tensors[cache_id]
    if case == "cache_rank":
        corrupted = cache.squeeze(0)
    elif case == "cache_sequence":
        corrupted = cache[:, :, :-1, :]
    elif case == "cache_dtype":
        corrupted = cache.to(torch.float32)
    elif case == "cache_nonfinite":
        corrupted = cache.clone()
        corrupted[0, 0, 0, 0] = float("inf")
    elif case == "cache_prefix_drift":
        corrupted = cache.clone()
        corrupted[0, 0, 0, 0] += 1
    else:
        corrupted = None
    if corrupted is not None:
        changed = _replace_capture_tensor(captured, "deep_tensors", cache_id, corrupted)
        return replace(source, captured=changed)

    if case == "layer0_append_mismatch":
        semantic_id = f"{prefix}.layer.00.kv.key"
        corrupted = captured.deep_tensors[semantic_id].clone()
        corrupted[0, 0, -1, 0] += 1
        changed = _replace_capture_tensor(
            captured, "deep_tensors", semantic_id, corrupted
        )
        return replace(source, captured=changed)

    if case == "layer0_value_append_mismatch":
        semantic_id = f"{prefix}.layer.00.kv.value"
        corrupted = captured.deep_tensors[semantic_id].clone()
        corrupted[0, 0, -1, 0] += 1
        changed = _replace_capture_tensor(
            captured, "deep_tensors", semantic_id, corrupted
        )
        return replace(source, captured=changed)

    mask_id = f"{prefix}.attention_mask"
    if case in {"mask_nonbinary", "mask_wrong_prefix", "mask_appended_zero"}:
        mask = captured.deep_tensors[mask_id].clone()
        if case == "mask_nonbinary":
            mask[0, 7] = 2
        elif case == "mask_wrong_prefix":
            mask[0, 7] = 0
        else:
            mask[0, -1] = 0
        changed = _replace_capture_tensor(captured, "deep_tensors", mask_id, mask)
        return replace(source, captured=changed)

    if case == "cache_position":
        changed = _replace_capture_tensor(
            captured,
            "deep_tensors",
            f"{prefix}.cache_position",
            torch.tensor([S - 1], dtype=torch.int64),
        )
        return replace(source, captured=changed)

    if case == "position_axis":
        semantic_id = f"{prefix}.position_ids"
        positions = captured.deep_tensors[semantic_id].clone()
        positions[1, 0, 0] += 1
        changed = _replace_capture_tensor(
            captured, "deep_tensors", semantic_id, positions
        )
        return replace(source, captured=changed)

    if case == "rope_delta":
        changed = _replace_capture_tensor(
            captured,
            "stage_tensors",
            "decoder.mrope.delta",
            torch.tensor([[-289]], dtype=torch.int64),
        )
        return replace(source, captured=changed)

    if case == "publication_case":
        return _replace_publication(source, case_id="ocr.other.0001")
    if case == "publication_text":
        return _replace_publication(source, decoded_text="JUX")
    if case == "publication_repeat3":
        return _replace_publication(source, repeat_count=3)
    if case == "publication_repeat_bool":
        publication = replace(source.publication, repeat_count=True)
        return replace(source, publication=publication)
    if case == "publication_repeat_nonint":
        publication = replace(source.publication, repeat_count="2")
        return replace(source, publication=publication)
    if case == "missing_trace":
        return replace(source, captured=replace(captured, token_trace=None))
    if case == "processor_terminal":
        semantic_id = "processor.input_ids"
        input_ids = captured.processor_tensors[semantic_id].clone()
        input_ids[0, -1] = 24
        changed = _replace_capture_tensor(
            captured, "processor_tensors", semantic_id, input_ids
        )
        return replace(source, captured=changed)
    if case == "prefill_logits_argmax":
        semantic_id = "decoder.prefill.logits.last"
        logits = captured.stage_tensors[semantic_id].clone()
        logits[0, PREFILL_TOKEN + 1] = 9.0
        changed = _replace_capture_tensor(
            captured, "stage_tensors", semantic_id, logits
        )
        return replace(source, captured=changed)
    if case == "mrope_terminal_position":
        semantic_id = "decoder.mrope.index"
        mrope_index = captured.stage_tensors[semantic_id].clone()
        mrope_index[1, 0, -1] = 40
        changed = _replace_capture_tensor(
            captured, "stage_tensors", semantic_id, mrope_index
        )
        return replace(source, captured=changed)

    trace = captured.token_trace
    assert trace is not None
    if case == "trace_token":
        bad_trace = replace(trace, tokens=(PREFILL_TOKEN, DECODE_TOKEN + 1))
        return replace(source, captured=replace(captured, token_trace=bad_trace))
    if case == "trace_input_chain":
        bad_step = replace(trace.steps[1], input_token=PREFILL_TOKEN + 1)
        bad_trace = replace(trace, steps=(trace.steps[0], bad_step))
        return replace(source, captured=replace(captured, token_trace=bad_trace))
    if case == "trace_kv_shapes":
        bad_step = replace(
            trace.steps[0], kv_shapes=((1, KVH, S - 1, D),) * L
        )
        bad_trace = replace(trace, steps=(bad_step, trace.steps[1]))
        return replace(source, captured=replace(captured, token_trace=bad_trace))
    if case == "alternate_tokens":
        alternate_prefill = PREFILL_TOKEN + 1
        alternate_decode = DECODE_TOKEN + 1
        first_step = replace(
            trace.steps[0],
            top_tokens=((alternate_prefill, 9.0), (PREFILL_TOKEN, 8.75)),
            chosen_token=alternate_prefill,
        )
        second_step = replace(
            trace.steps[1],
            input_token=alternate_prefill,
            top_tokens=((alternate_decode, 9.0), (DECODE_TOKEN, 8.5)),
            chosen_token=alternate_decode,
        )
        alternate_trace = replace(
            trace,
            tokens=(alternate_prefill, alternate_decode),
            steps=(first_step, second_step),
        )
        prefill_logits = captured.stage_tensors[
            "decoder.prefill.logits.last"
        ].clone()
        prefill_logits[0, PREFILL_TOKEN] = 0
        prefill_logits[0, alternate_prefill] = 9.0
        decode_logits_id = f"{prefix}.logits"
        decode_logits = captured.deep_tensors[decode_logits_id].clone()
        decode_logits[0, 0, DECODE_TOKEN] = 0
        decode_logits[0, 0, alternate_decode] = 9.0
        stage = dict(captured.stage_tensors)
        stage["decoder.prefill.logits.last"] = prefill_logits
        deep = dict(captured.deep_tensors)
        deep[decode_logits_id] = decode_logits
        alternate_capture = replace(
            captured,
            stage_tensors=stage,
            deep_tensors=deep,
            token_trace=alternate_trace,
        )
        alternate_source = _replace_publication(
            source, generated_tokens=(alternate_prefill, alternate_decode)
        )
        return replace(alternate_source, captured=alternate_capture)

    if case == "logits_argmax":
        semantic_id = f"{prefix}.logits"
        logits = captured.deep_tensors[semantic_id].clone()
        logits[0, 0, DECODE_TOKEN + 1] = 9.0
        changed = _replace_capture_tensor(
            captured, "deep_tensors", semantic_id, logits
        )
        return replace(source, captured=changed)

    pipeline_id = f"{prefix}.layer.00.q"
    if case == "pipeline_shape":
        pipeline = captured.deep_tensors[pipeline_id][..., :-1]
    elif case == "pipeline_dtype":
        pipeline = captured.deep_tensors[pipeline_id].to(torch.float32)
    elif case == "pipeline_nonfinite":
        pipeline_id = f"{prefix}.final_norm"
        pipeline = captured.deep_tensors[pipeline_id].clone()
        pipeline[0, 0, 0] = float("nan")
    else:
        pipeline = None
    if pipeline is not None:
        changed = _replace_capture_tensor(
            captured, "deep_tensors", pipeline_id, pipeline
        )
        return replace(source, captured=changed)

    logits_id = f"{prefix}.logits"
    if case == "logits_shape":
        logits = captured.deep_tensors[logits_id][..., :-1]
    elif case == "logits_dtype":
        logits = captured.deep_tensors[logits_id].to(torch.float32)
    elif case == "logits_nonfinite":
        logits = captured.deep_tensors[logits_id].clone()
        logits[0, 0, 0] = float("inf")
    else:
        logits = None
    if logits is not None:
        changed = _replace_capture_tensor(
            captured, "deep_tensors", logits_id, logits
        )
        return replace(source, captured=changed)

    if case == "provenance_device":
        return replace(source, provenance=replace(source.provenance, device="cpu"))
    if case == "provenance_dtype":
        return replace(source, provenance=replace(source.provenance, dtype="float32"))
    if case == "provenance_revision":
        return replace(
            source,
            provenance=replace(source.provenance, model_revision="0" * 40),
        )
    if case == "provenance_version":
        return replace(
            source,
            provenance=replace(source.provenance, trace_schema_version=2),
        )
    if case == "provenance_model_lock":
        return replace(
            source,
            provenance=replace(
                source.provenance, model_lock_hash=f"blake3:{'0' * 64}"
            ),
        )
    if case == "provenance_capture_tool":
        return replace(
            source,
            provenance=replace(source.provenance, capture_tool_version="0.2.0"),
        )
    if case == "provenance_python":
        return replace(
            source,
            provenance=replace(source.provenance, python_version="3.13.0"),
        )
    if case == "provenance_torch":
        return replace(
            source,
            provenance=replace(source.provenance, torch_version="2.14.0"),
        )
    if case == "provenance_transformers":
        return replace(
            source,
            provenance=replace(source.provenance, transformers_version="5.15.0"),
        )
    if case == "provenance_shim":
        return replace(
            source,
            provenance=replace(
                source.provenance,
                compatibility_shims=("paddleocr-vl-1.6/transformers-v5-abi@2",),
            ),
        )
    if case == "provenance_nondeterministic":
        return replace(
            source,
            provenance=replace(source.provenance, deterministic_algorithms=False),
        )
    if case == "repeat_count":
        return replace(
            source, publication=replace(source.publication, repeat_count=1)
        )
    if case == "source_digest":
        return replace(
            source,
            publication=replace(source.publication, bundle_digest="blake3:short"),
        )
    if case == "source_fingerprint":
        return replace(
            source,
            publication=replace(
                source.publication, semantic_fingerprint=f"sha256:{'b' * 64}"
            ),
        )
    if case == "publication_lock_digest_malformed":
        return replace(source, publication_lock_blake3="blake3:short")
    if case == "publication_lock_digest_wrong":
        return replace(source, publication_lock_blake3=f"blake3:{'e' * 64}")
    if case == "publication_stale_lock_after_fingerprint":
        return replace(
            source,
            publication=replace(
                source.publication,
                semantic_fingerprint=f"blake3:{'f' * 64}",
            ),
        )
    raise AssertionError(f"unknown corruption case: {case}")


def _malform_source_tensor(
    source,
    mapping_name: str,
    semantic_id: str,
    mutation: str,
):
    tensor = getattr(source.captured, mapping_name)[semantic_id]
    if mutation == "shape":
        corrupted = tensor[..., :-1]
    elif mutation == "dtype":
        target_dtype = torch.int32 if tensor.dtype == torch.int64 else torch.float32
        corrupted = tensor.to(target_dtype)
    elif mutation == "nonfinite":
        assert tensor.is_floating_point()
        corrupted = tensor.clone()
        corrupted.reshape(-1)[0] = float("inf")
    elif mutation == "cache_rank":
        corrupted = tensor.squeeze(0)
    elif mutation == "cache_sequence":
        corrupted = tensor[:, :, :-1, :]
    elif mutation == "invalid_value":
        corrupted = tensor.clone()
        if semantic_id.endswith(".attention_mask"):
            corrupted[0, 0] = 2
        elif semantic_id.endswith(".cache_position"):
            corrupted[0] = S - 1
        elif semantic_id.endswith(".position_ids"):
            corrupted[1, 0, 0] += 1
        elif semantic_id == "decoder.mrope.delta":
            corrupted[0, 0] = -289
        else:
            raise AssertionError(f"no invalid value mutation for {semantic_id}")
    elif mutation == "prefix_drift":
        corrupted = tensor.clone()
        corrupted[0, 0, 0, 0] += 1
    else:
        raise AssertionError(f"unknown tensor mutation: {mutation}")

    changed = _replace_capture_tensor(
        source.captured,
        mapping_name,
        semantic_id,
        corrupted,
    )
    return replace(source, captured=changed)


def _tensor_contract_cases() -> tuple[tuple[str, str, str, str, str], ...]:
    cases: list[tuple[str, str, str, str, str]] = []

    def case_name(semantic_id: str, mutation: str) -> str:
        return f"{semantic_id.replace('.', '-')}-{mutation}"

    def add_float_contract(mapping_name: str, semantic_id: str) -> None:
        for mutation, expected_code in (
            ("shape", "invalid_tensor_shape"),
            ("dtype", "invalid_tensor_dtype"),
            ("nonfinite", "nonfinite_tensor"),
        ):
            cases.append(
                (
                    case_name(semantic_id, mutation),
                    mapping_name,
                    semantic_id,
                    mutation,
                    expected_code,
                )
            )

    prefix = "decoder.decode.00"
    for semantic_id in (
        f"{prefix}.rope.cos",
        f"{prefix}.rope.sin",
        *(f"{prefix}.layer.{layer_index:02d}.output" for layer_index in range(L)),
        f"{prefix}.layer.00.input",
        f"{prefix}.layer.00.norm1",
        f"{prefix}.layer.00.q",
        f"{prefix}.layer.00.k",
        f"{prefix}.layer.00.v",
        f"{prefix}.layer.00.mrope.q",
        f"{prefix}.layer.00.mrope.k",
        f"{prefix}.layer.00.attention.context",
        f"{prefix}.layer.00.attention.output",
        f"{prefix}.layer.00.attention.residual",
        f"{prefix}.layer.00.norm2",
        f"{prefix}.layer.00.mlp.gate",
        f"{prefix}.layer.00.mlp.up",
        f"{prefix}.layer.00.mlp.activation",
        f"{prefix}.layer.00.mlp.down",
        f"{prefix}.final_norm",
        f"{prefix}.logits",
        f"{prefix}.layer.03.kv.value",
    ):
        add_float_contract("deep_tensors", semantic_id)

    for mapping_name, semantic_id, value_code in (
        ("deep_tensors", f"{prefix}.attention_mask", "invalid_attention_mask"),
        ("deep_tensors", f"{prefix}.cache_position", "cache_position_mismatch"),
        ("deep_tensors", f"{prefix}.position_ids", "position_id_mismatch"),
        ("stage_tensors", "decoder.mrope.delta", "rope_delta_mismatch"),
    ):
        for mutation, expected_code in (
            ("shape", "invalid_tensor_shape"),
            ("dtype", "invalid_tensor_dtype"),
            ("invalid_value", value_code),
        ):
            cases.append(
                (
                    case_name(semantic_id, mutation),
                    mapping_name,
                    semantic_id,
                    mutation,
                    expected_code,
                )
            )

    for kind in ("key", "value"):
        semantic_id = f"decoder.layer.07.kv.{kind}"
        for mutation, expected_code in (
            ("cache_rank", "invalid_tensor_shape"),
            ("cache_sequence", "invalid_tensor_shape"),
            ("dtype", "invalid_tensor_dtype"),
            ("nonfinite", "nonfinite_tensor"),
        ):
            cases.append(
                (
                    case_name(semantic_id, mutation),
                    "deep_tensors",
                    semantic_id,
                    mutation,
                    expected_code,
                )
            )

    value_cache_id = f"{prefix}.layer.03.kv.value"
    cases.append(
        (
            case_name(value_cache_id, "prefix_drift"),
            "deep_tensors",
            value_cache_id,
            "prefix_drift",
            "cache_prefix_mismatch",
        )
    )
    assert len(cases) == 135
    assert len({case[0] for case in cases}) == len(cases)
    return tuple(cases)


TENSOR_CONTRACT_CASES = _tensor_contract_cases()


@lru_cache(maxsize=1)
def _tensor_contract_source():
    api = importlib.import_module("pvlc_reference.decode_fixture")
    return _make_source(api, seed=20260719)


def _top_tokens_from_logits(
    logits: torch.Tensor,
    *,
    count: int,
) -> tuple[tuple[int, float], ...]:
    ranked = sorted(
        (
            (token_id, float(score))
            for token_id, score in enumerate(logits.tolist())
        ),
        key=lambda item: (-item[1], item[0]),
    )
    return tuple(ranked[:count])


def _make_tied_top_tokens_source(api):
    source = _make_source(api, seed=20260719)
    captured = source.captured
    trace = captured.token_trace
    assert trace is not None

    prefill_tied_tokens = (111, 222)
    decode_tied_tokens = (333, 444)
    prefill_tied_score = 7.5
    decode_tied_score = 8.1875

    prefill_logits_id = "decoder.prefill.logits.last"
    prefill_logits = captured.stage_tensors[prefill_logits_id].clone()
    prefill_logits.fill_(-10)
    prefill_logits[0, PREFILL_TOKEN] = 8.75
    for token_id in prefill_tied_tokens:
        prefill_logits[0, token_id] = prefill_tied_score

    decode_logits_id = "decoder.decode.00.logits"
    decode_logits = captured.deep_tensors[decode_logits_id].clone()
    decode_logits.fill_(-10)
    decode_logits[0, 0, DECODE_TOKEN] = 8.5
    for token_id in decode_tied_tokens:
        decode_logits[0, 0, token_id] = decode_tied_score

    prefill_top_tokens = _top_tokens_from_logits(prefill_logits[0], count=3)
    decode_top_tokens = _top_tokens_from_logits(decode_logits[0, 0], count=3)
    assert prefill_top_tokens == (
        (PREFILL_TOKEN, 8.75),
        (prefill_tied_tokens[0], prefill_tied_score),
        (prefill_tied_tokens[1], prefill_tied_score),
    )
    assert decode_top_tokens == (
        (DECODE_TOKEN, 8.5),
        (decode_tied_tokens[0], decode_tied_score),
        (decode_tied_tokens[1], decode_tied_score),
    )

    stage = dict(captured.stage_tensors)
    stage[prefill_logits_id] = prefill_logits
    deep = dict(captured.deep_tensors)
    deep[decode_logits_id] = decode_logits
    tied_trace = replace(
        trace,
        steps=(
            replace(trace.steps[0], top_tokens=prefill_top_tokens),
            replace(trace.steps[1], top_tokens=decode_top_tokens),
        ),
    )
    tied_capture = replace(
        captured,
        stage_tensors=stage,
        deep_tensors=deep,
        token_trace=tied_trace,
    )
    tied_source = replace(source, captured=tied_capture)
    assert tied_source.publication.generated_tokens == (PREFILL_TOKEN, DECODE_TOKEN)
    assert tied_trace.tokens == tied_source.publication.generated_tokens
    assert tied_trace.steps[0].chosen_token == PREFILL_TOKEN
    assert tied_trace.steps[1].chosen_token == DECODE_TOKEN
    return tied_source


def _make_boundary_tied_top_tokens_source(api):
    source = _make_source(api, seed=20260719)
    captured = source.captured
    trace = captured.token_trace
    assert trace is not None

    prefill_excluded_tied_token = 110
    prefill_selected_tied_tokens = (111, 222)
    decode_excluded_tied_token = 332
    decode_selected_tied_tokens = (333, 444)
    prefill_tied_score = 7.5
    decode_tied_score = 8.1875

    prefill_logits_id = "decoder.prefill.logits.last"
    prefill_logits = captured.stage_tensors[prefill_logits_id].clone()
    prefill_logits.fill_(-10)
    prefill_logits[0, PREFILL_TOKEN] = 8.75
    for token_id in (
        prefill_excluded_tied_token,
        *prefill_selected_tied_tokens,
    ):
        prefill_logits[0, token_id] = prefill_tied_score

    decode_logits_id = "decoder.decode.00.logits"
    decode_logits = captured.deep_tensors[decode_logits_id].clone()
    decode_logits.fill_(-10)
    decode_logits[0, 0, DECODE_TOKEN] = 8.5
    for token_id in (
        decode_excluded_tied_token,
        *decode_selected_tied_tokens,
    ):
        decode_logits[0, 0, token_id] = decode_tied_score

    prefill_trace_top_tokens = (
        (PREFILL_TOKEN, 8.75),
        (prefill_selected_tied_tokens[0], prefill_tied_score),
        (prefill_selected_tied_tokens[1], prefill_tied_score),
    )
    decode_trace_top_tokens = (
        (DECODE_TOKEN, 8.5),
        (decode_selected_tied_tokens[0], decode_tied_score),
        (decode_selected_tied_tokens[1], decode_tied_score),
    )
    assert _top_tokens_from_logits(prefill_logits[0], count=3) == (
        (PREFILL_TOKEN, 8.75),
        (prefill_excluded_tied_token, prefill_tied_score),
        (prefill_selected_tied_tokens[0], prefill_tied_score),
    )
    assert _top_tokens_from_logits(decode_logits[0, 0], count=3) == (
        (DECODE_TOKEN, 8.5),
        (decode_excluded_tied_token, decode_tied_score),
        (decode_selected_tied_tokens[0], decode_tied_score),
    )

    stage = dict(captured.stage_tensors)
    stage[prefill_logits_id] = prefill_logits
    deep = dict(captured.deep_tensors)
    deep[decode_logits_id] = decode_logits
    boundary_trace = replace(
        trace,
        steps=(
            replace(trace.steps[0], top_tokens=prefill_trace_top_tokens),
            replace(trace.steps[1], top_tokens=decode_trace_top_tokens),
        ),
    )
    boundary_capture = replace(
        captured,
        stage_tensors=stage,
        deep_tensors=deep,
        token_trace=boundary_trace,
    )
    boundary_source = replace(source, captured=boundary_capture)
    return (
        boundary_source,
        {
            "prefill_excluded_tied_token": prefill_excluded_tied_token,
            "prefill_selected_tied_tokens": prefill_selected_tied_tokens,
            "prefill_tied_score": prefill_tied_score,
            "decode_excluded_tied_token": decode_excluded_tied_token,
            "decode_selected_tied_tokens": decode_selected_tied_tokens,
            "decode_tied_score": decode_tied_score,
        },
    )


def _patch_topk_with_boundary_selected_subset(
    api,
    monkeypatch: pytest.MonkeyPatch,
    *,
    prefill_logits: torch.Tensor,
    decode_logits: torch.Tensor,
    prefill_selected_tied_tokens: tuple[int, int],
    decode_selected_tied_tokens: tuple[int, int],
) -> None:
    original_topk = api.torch.topk

    def boundary_selected_subset_topk(logits, k, *args, **kwargs):
        if logits.data_ptr() == prefill_logits.data_ptr() and k == 3:
            return (
                torch.tensor(
                    [8.75, 7.5, 7.5],
                    dtype=logits.dtype,
                    device=logits.device,
                ),
                torch.tensor(
                    [PREFILL_TOKEN, *sorted(prefill_selected_tied_tokens, reverse=True)],
                    dtype=torch.int64,
                    device=logits.device,
                ),
            )
        if logits.data_ptr() == decode_logits.data_ptr() and k == 3:
            return (
                torch.tensor(
                    [8.5, 8.1875, 8.1875],
                    dtype=logits.dtype,
                    device=logits.device,
                ),
                torch.tensor(
                    [DECODE_TOKEN, *sorted(decode_selected_tied_tokens, reverse=True)],
                    dtype=torch.int64,
                    device=logits.device,
                ),
            )
        return original_topk(logits, k, *args, **kwargs)

    monkeypatch.setattr(api.torch, "topk", boundary_selected_subset_topk)


@pytest.fixture(scope="module")
def verified_decode_source_bundles(tmp_path_factory):
    output_root = tmp_path_factory.mktemp("decode-fixture-loader")
    case = CaseSpec.load(SMOKE_CASE_PATH)
    source_image = SMOKE_IMAGE_PATH.read_bytes()
    captured, provenance = synthetic_capture(seed=20260719)
    assert captured.token_trace is not None
    lower_ranked_trace = replace(
        captured.token_trace,
        steps=(
            replace(
                captured.token_trace.steps[0],
                top_tokens=(
                    (PREFILL_TOKEN, 8.75),
                    (12345, 6.25),
                    (77, -0.5),
                ),
            ),
            replace(
                captured.token_trace.steps[1],
                top_tokens=(
                    (DECODE_TOKEN, 8.5),
                    (23456, 7.0),
                    (88, -1.25),
                ),
            ),
        ),
    )
    lower_ranked_stage = dict(captured.stage_tensors)
    lower_ranked_prefill_logits = lower_ranked_stage[
        "decoder.prefill.logits.last"
    ].clone()
    lower_ranked_prefill_logits.fill_(-10)
    lower_ranked_prefill_logits[0, PREFILL_TOKEN] = 8.75
    lower_ranked_prefill_logits[0, 12345] = 6.25
    lower_ranked_prefill_logits[0, 77] = -0.5
    lower_ranked_stage["decoder.prefill.logits.last"] = lower_ranked_prefill_logits

    lower_ranked_deep = dict(captured.deep_tensors)
    lower_ranked_decode_logits = lower_ranked_deep[
        "decoder.decode.00.logits"
    ].clone()
    lower_ranked_decode_logits.fill_(-10)
    lower_ranked_decode_logits[0, 0, DECODE_TOKEN] = 8.5
    lower_ranked_decode_logits[0, 0, 23456] = 7.0
    lower_ranked_decode_logits[0, 0, 88] = -1.25
    lower_ranked_deep["decoder.decode.00.logits"] = lower_ranked_decode_logits
    lower_ranked_capture = replace(
        captured,
        stage_tensors=lower_ranked_stage,
        deep_tensors=lower_ranked_deep,
        token_trace=lower_ranked_trace,
    )
    assert _top_tokens_from_logits(
        lower_ranked_prefill_logits[0], count=3
    ) == lower_ranked_trace.steps[0].top_tokens
    assert _top_tokens_from_logits(
        lower_ranked_decode_logits[0, 0], count=3
    ) == lower_ranked_trace.steps[1].top_tokens

    def build(
        name: str,
        *,
        trace_level: TraceLevel,
        bundle_case: CaseSpec = case,
        bundle_provenance: CaptureProvenance = provenance,
        bundle_captured: CapturedArtifacts = captured,
    ):
        root = output_root / name
        result = export_golden_bundle(
            root=root,
            case=bundle_case,
            source_image=source_image,
            provenance=bundle_provenance,
            trace_level=trace_level,
            captured=bundle_captured,
            probe_seed=20260719,
        )
        publication = GoldenEntry(
            case_id=bundle_case.case_id,
            trace_level=trace_level,
            artifact_path=root.name,
            bundle_digest=result.bundle_digest,
            semantic_fingerprint=OFFICIAL_SEMANTIC_FINGERPRINT,
            generated_tokens=(PREFILL_TOKEN, DECODE_TOKEN),
            decoded_text="JUL",
            repeat_count=2,
        )
        publication_lock = _publication_lock(publication)
        lock_path = output_root / f"{name}.golden.lock"
        lock_bytes = publication_lock.canonical_bytes()
        lock_path.write_bytes(lock_bytes)
        lock_digest = f"blake3:{blake3(lock_bytes).hexdigest()}"
        return root, result, publication, lock_path, lock_digest

    l3 = build("official-l3", trace_level=TraceLevel.L3)
    l2 = build("official-l2", trace_level=TraceLevel.L2)
    wrong_case = build(
        "wrong-case-l3",
        trace_level=TraceLevel.L3,
        bundle_case=replace(case, case_id="ocr.not_official.0001"),
    )
    wrong_device = build(
        "wrong-device-l3",
        trace_level=TraceLevel.L3,
        bundle_provenance=replace(provenance, device="cpu"),
    )
    wrong_dtype = build(
        "wrong-dtype-l3",
        trace_level=TraceLevel.L3,
        bundle_provenance=replace(provenance, dtype="float32"),
    )
    lower_ranked = build(
        "lower-ranked-top-tokens-l3",
        trace_level=TraceLevel.L3,
        bundle_captured=lower_ranked_capture,
    )
    return {
        "captured": captured,
        "provenance": provenance,
        "case": case,
        "l3": l3,
        "l2": l2,
        "wrong_case": wrong_case,
        "wrong_device": wrong_device,
        "wrong_dtype": wrong_dtype,
        "lower_ranked": lower_ranked,
        "lower_ranked_capture": lower_ranked_capture,
    }


def _assert_captured_tensors_exact(
    actual: CapturedArtifacts,
    expected: CapturedArtifacts,
) -> None:
    for group_name in ("processor_tensors", "stage_tensors", "deep_tensors"):
        actual_group = getattr(actual, group_name)
        expected_group = getattr(expected, group_name)
        assert set(actual_group) == set(expected_group)
        for semantic_id, expected_tensor in expected_group.items():
            actual_tensor = actual_group[semantic_id]
            assert actual_tensor.dtype == expected_tensor.dtype
            assert actual_tensor.shape == expected_tensor.shape
            assert torch.equal(actual_tensor, expected_tensor)


def _bundle_bytes_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _copy_locked_bundle(
    tmp_path: Path,
    bundle_root: Path,
    publication: GoldenEntry,
) -> tuple[Path, GoldenEntry, Path, str]:
    local_bundle = tmp_path / publication.artifact_path
    local_bundle.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(bundle_root, local_bundle)
    local_publication = replace(
        publication,
        artifact_path=local_bundle.relative_to(tmp_path).as_posix(),
    )
    lock_path = tmp_path / "golden.lock"
    lock_bytes = _publication_lock(local_publication).canonical_bytes()
    lock_path.write_bytes(lock_bytes)
    lock_digest = f"blake3:{blake3(lock_bytes).hexdigest()}"
    return local_bundle, local_publication, lock_path, lock_digest


def _resolve_open_target(
    path: str | os.PathLike[str],
    *,
    dir_fd: int | None,
) -> Path | None:
    candidate = Path(os.fsdecode(os.fspath(path)))
    if candidate.is_absolute():
        return Path(os.path.abspath(candidate))
    if dir_fd is None:
        return Path(os.path.abspath(candidate))
    try:
        directory = Path(os.readlink(f"/dev/fd/{dir_fd}"))
    except OSError:
        return None
    return Path(os.path.abspath(directory / candidate))


def _assert_open_flags_include_security_bits(flags: int) -> None:
    for name in ("O_NOFOLLOW", "O_NONBLOCK", "O_CLOEXEC"):
        value = getattr(os, name, 0)
        if value:
            assert flags & value == value


def _make_bundle_relative_open_observer(
    local_bundle: Path,
    artifact_name: str,
):
    expected_bundle_path = Path(os.path.abspath(local_bundle))
    expected_artifact_path = Path(os.path.abspath(local_bundle / artifact_name))
    expected_bundle_identity = _stat_identity(
        os.stat(local_bundle, follow_symlinks=False)
    )
    state = {
        "bundle_dir_fd": None,
        "bundle_dir_open_count": 0,
        "artifact_open_count": 0,
    }

    def observe(
        path: str | os.PathLike[str],
        descriptor: int,
        *,
        dir_fd: int | None,
        verified: bool,
    ) -> bool:
        candidate = Path(os.fsdecode(os.fspath(path)))
        descriptor_stat = os.fstat(descriptor)
        resolved_absolute = None
        if candidate.is_absolute():
            resolved_absolute = Path(os.path.abspath(candidate))
            if (
                resolved_absolute == expected_bundle_path
                and stat.S_ISDIR(descriptor_stat.st_mode)
                and _stat_identity(descriptor_stat) == expected_bundle_identity
            ):
                state["bundle_dir_open_count"] += 1
                if state["bundle_dir_fd"] is None:
                    state["bundle_dir_fd"] = descriptor

        if not verified:
            return False

        if (
            not candidate.is_absolute()
            and candidate == Path(artifact_name)
            and dir_fd == state["bundle_dir_fd"]
        ):
            state["artifact_open_count"] += 1
            return True

        if state["bundle_dir_fd"] is None and resolved_absolute == expected_artifact_path:
            state["artifact_open_count"] += 1
            return True

        return False

    return state, observe


def _supports_alarm_guard() -> bool:
    return all(
        hasattr(signal, name)
        for name in ("SIGALRM", "setitimer", "ITIMER_REAL")
    )


@contextmanager
def _alarm_guard(label: str):
    if not _supports_alarm_guard():
        pytest.skip("alarm guard requires SIGALRM and setitimer support")

    class _AlarmFired(RuntimeError):
        pass

    state = {"fired": False}

    def alarm_handler(_signum, _frame):
        state["fired"] = True
        raise _AlarmFired(f"emergency guard fired for {label}")

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, 0.5)
        yield state
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _rebuild_bundle_hashes(root: Path) -> str:
    manifest_path = root / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(json.loads(manifest_path.read_bytes())))
    artifacts: dict[str, dict[str, int | str]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "hashes.json":
            continue
        payload = path.read_bytes()
        artifacts[path.relative_to(root).as_posix()] = {
            "blake3": blake3(payload).hexdigest(),
            "size": len(payload),
        }
    hashes_bytes = canonical_json_bytes(
        {"algorithm": "blake3", "artifacts": artifacts, "format_version": 1}
    )
    (root / "hashes.json").write_bytes(hashes_bytes)
    return f"blake3:{blake3(hashes_bytes).hexdigest()}"


def _rewrite_safetensors_value(path: Path, semantic_id: str) -> None:
    tensors = {name: tensor.clone() for name, tensor in load_file(path).items()}
    tensor = tensors[semantic_id]
    replacement = 17.0 if float(tensor.reshape(-1)[0].item()) != 17.0 else 18.0
    tensor.reshape(-1)[0] = replacement
    path.write_bytes(serialize_safetensors(tensors))


def test_synthetic_capture_is_self_consistent() -> None:
    assert blake3(M5_LAYER0_FIXTURE_PATH.read_bytes()).hexdigest() == (
        "30eed2f7a4d9336f8b3429d7294065c45214c4425f7559e8e2059ebd54c89522"
    )
    captured, provenance = synthetic_capture(seed=20260719)
    provenance.validate()
    assert captured.token_trace is not None
    captured.token_trace.validate()
    assert captured.token_trace.tokens == (PREFILL_TOKEN, DECODE_TOKEN)

    prefix = "decoder.decode.00"
    assert captured.processor_tensors["processor.input_ids"].shape == (1, S)
    assert captured.stage_tensors["decoder.prefill.logits.last"].argmax().item() == PREFILL_TOKEN
    assert captured.deep_tensors[f"{prefix}.logits"].argmax().item() == DECODE_TOKEN
    assert torch.equal(
        captured.deep_tensors[f"{prefix}.attention_mask"][:, :-1],
        captured.processor_tensors["processor.attention_mask"],
    )

    for layer_index in range(L):
        layer = f"{layer_index:02d}"
        for kind in ("key", "value"):
            prefill = captured.deep_tensors[f"decoder.layer.{layer}.kv.{kind}"]
            post = captured.deep_tensors[f"{prefix}.layer.{layer}.kv.{kind}"]
            assert prefill.shape == (1, KVH, S, D)
            assert post.shape == (1, KVH, S + 1, D)
            assert torch.equal(post[:, :, :S, :], prefill)
        assert captured.deep_tensors[f"{prefix}.layer.{layer}.output"].shape == (
            1,
            1,
            H,
        )

    deep = captured.deep_tensors
    assert torch.equal(
        deep[f"{prefix}.layer.00.attention.residual"],
        deep[f"{prefix}.layer.00.input"]
        + deep[f"{prefix}.layer.00.attention.output"],
    )
    assert torch.equal(
        deep[f"{prefix}.layer.00.output"],
        deep[f"{prefix}.layer.00.attention.residual"]
        + deep[f"{prefix}.layer.00.mlp.down"],
    )
    assert torch.equal(
        deep[f"{prefix}.layer.00.kv.key"][:, :, -1:, :],
        deep[f"{prefix}.layer.00.mrope.k"],
    )
    raw_q = (
        deep[f"{prefix}.layer.00.q"]
        .view(1, 1, QH, D)
        .transpose(1, 2)
        .contiguous()
    )
    raw_k = (
        deep[f"{prefix}.layer.00.k"]
        .view(1, 1, KVH, D)
        .transpose(1, 2)
        .contiguous()
    )
    assert not torch.equal(raw_q, deep[f"{prefix}.layer.00.mrope.q"])
    assert not torch.equal(raw_k, deep[f"{prefix}.layer.00.mrope.k"])
    assert deep[f"{prefix}.final_norm"].shape == (1, 1, H)
    assert deep[f"{prefix}.logits"].shape == (1, 1, V)


def test_serialize_decode_fixture_is_deterministic_isolated_and_immutable(
    tmp_path: Path,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    source = _make_source(api, seed=20260719)

    def reversed_mapping(mapping) -> dict[str, torch.Tensor]:
        return dict(reversed(tuple(mapping.items())))

    reordered_capture = replace(
        source.captured,
        processor_tensors=reversed_mapping(source.captured.processor_tensors),
        stage_tensors=reversed_mapping(source.captured.stage_tensors),
        deep_tensors=reversed_mapping(source.captured.deep_tensors),
    )
    reordered_source = replace(source, captured=reordered_capture)

    extra_deep = dict(source.captured.deep_tensors)
    irrelevant_id = "vision.future.unselected.verified_source_tensor"
    extra_deep[irrelevant_id] = torch.tensor([17.0], dtype=torch.bfloat16)
    isolated_capture = replace(source.captured, deep_tensors=extra_deep)
    isolated_source = replace(source, captured=isolated_capture)
    republished_source = _replace_publication(
        isolated_source, semantic_fingerprint=f"blake3:{'d' * 64}"
    )
    different_source = _replace_publication(
        _make_source(api, seed=20260720),
        semantic_fingerprint=f"blake3:{'c' * 64}",
    )
    assert republished_source.publication_lock_blake3 == _publication_lock_blake3(
        republished_source.publication
    )
    assert republished_source.publication_lock_blake3 != source.publication_lock_blake3

    mapping_orders = tuple(
        (mapping, tuple(mapping))
        for captured in (
            source.captured,
            reordered_capture,
            isolated_capture,
            different_source.captured,
        )
        for mapping in (
            captured.processor_tensors,
            captured.stage_tensors,
            captured.deep_tensors,
        )
    )
    tensor_snapshots = _snapshot_source_tensors(
        source.captured,
        reordered_capture,
        isolated_capture,
        different_source.captured,
    )

    first = api.serialize_decode_fixture(source)
    repeated = api.serialize_decode_fixture(source)
    reordered = api.serialize_decode_fixture(reordered_source)
    isolated = api.serialize_decode_fixture(isolated_source)
    republished = api.serialize_decode_fixture(republished_source)
    different = api.serialize_decode_fixture(different_source)

    assert first == repeated == reordered == isolated
    assert republished != first
    assert different != first
    for mapping, original_order in mapping_orders:
        assert tuple(mapping) == original_order
    for tensor, snapshot in tensor_snapshots:
        assert tensor.dtype == snapshot.dtype
        assert tensor.shape == snapshot.shape
        assert torch.equal(tensor, snapshot)

    first_path = tmp_path / "seed-1.safetensors"
    republished_path = tmp_path / "seed-1-republished.safetensors"
    different_path = tmp_path / "seed-2.safetensors"
    first_path.write_bytes(first)
    republished_path.write_bytes(republished)
    different_path.write_bytes(different)
    with safe_open(first_path, framework="pt", device="cpu") as reader:
        assert len(reader.keys()) == 44
        assert irrelevant_id not in reader.keys()
        first_metadata = reader.metadata()
    with safe_open(republished_path, framework="pt", device="cpu") as reader:
        assert len(reader.keys()) == 44
        assert irrelevant_id not in reader.keys()
        republished_metadata = reader.metadata()
    with safe_open(different_path, framework="pt", device="cpu") as reader:
        assert len(reader.keys()) == 44
        assert reader.metadata()["source_semantic_fingerprint"] == f"blake3:{'c' * 64}"
    assert first_metadata is not None
    assert republished_metadata is not None
    assert set(first_metadata) == set(republished_metadata)
    assert {
        key
        for key in first_metadata
        if first_metadata[key] != republished_metadata[key]
    } == {"source_semantic_fingerprint", "source_publication_lock_blake3"}
    assert first_metadata["source_semantic_fingerprint"] == OFFICIAL_SEMANTIC_FINGERPRINT
    assert republished_metadata["source_semantic_fingerprint"] == f"blake3:{'d' * 64}"
    assert first_metadata["source_publication_lock_blake3"] == (
        source.publication_lock_blake3
    )
    assert republished_metadata["source_publication_lock_blake3"] == (
        republished_source.publication_lock_blake3
    )

    first_tensors = load_file(first_path, device="cpu")
    republished_tensors = load_file(republished_path, device="cpu")
    different_tensors = load_file(different_path, device="cpu")
    assert len(first_tensors) == len(republished_tensors) == len(different_tensors) == 44
    assert set(first_tensors) == set(republished_tensors)
    for semantic_id in first_tensors:
        assert torch.equal(
            first_tensors[semantic_id].view(torch.uint8),
            republished_tensors[semantic_id].view(torch.uint8),
        )
    for selected_source, actual in (
        (source, first_tensors),
        (different_source, different_tensors),
    ):
        expected = _expected_output_tensors(selected_source)
        assert set(actual) == set(expected)
        for semantic_id, expected_tensor in expected.items():
            assert actual[semantic_id].dtype == expected_tensor.dtype
            assert actual[semantic_id].shape == expected_tensor.shape
            assert torch.equal(
                actual[semantic_id].view(torch.uint8),
                expected_tensor.contiguous().view(torch.uint8),
            )


def test_serialize_decode_fixture_accepts_canonical_tied_top_tokens_independent_of_backend_topk_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    source = _make_tied_top_tokens_source(api)
    prefill_logits = source.captured.stage_tensors["decoder.prefill.logits.last"][0]
    decode_logits = source.captured.deep_tensors["decoder.decode.00.logits"][0, 0]
    assert _top_tokens_from_logits(prefill_logits, count=3) == source.captured.token_trace.steps[
        0
    ].top_tokens
    assert _top_tokens_from_logits(decode_logits, count=3) == source.captured.token_trace.steps[
        1
    ].top_tokens
    assert int(torch.argmax(prefill_logits).item()) == PREFILL_TOKEN
    assert int(torch.argmax(decode_logits).item()) == DECODE_TOKEN
    assert source.publication.generated_tokens == (PREFILL_TOKEN, DECODE_TOKEN)

    original_topk = api.torch.topk

    def descending_tie_topk(logits, k, *args, **kwargs):
        scores, token_ids = original_topk(logits, k, *args, **kwargs)
        tied_descending = sorted(
            zip(token_ids.tolist(), scores.tolist(), strict=True),
            key=lambda item: (-float(item[1]), -int(item[0])),
        )
        reordered_scores = torch.tensor(
            [score for _, score in tied_descending],
            dtype=scores.dtype,
            device=scores.device,
        )
        reordered_token_ids = torch.tensor(
            [token_id for token_id, _ in tied_descending],
            dtype=token_ids.dtype,
            device=token_ids.device,
        )
        return reordered_scores, reordered_token_ids

    monkeypatch.setattr(api.torch, "topk", descending_tie_topk)
    patched_prefill_scores, patched_prefill_token_ids = api.torch.topk(prefill_logits, k=3)
    canonical_prefill_token_ids = tuple(
        token_id for token_id, _ in source.captured.token_trace.steps[0].top_tokens
    )
    assert patched_prefill_scores.tolist()[1] == patched_prefill_scores.tolist()[2]
    assert tuple(int(token_id) for token_id in patched_prefill_token_ids.tolist()[1:]) == tuple(
        sorted(canonical_prefill_token_ids[1:], reverse=True)
    )
    assert tuple(int(token_id) for token_id in patched_prefill_token_ids.tolist()) != (
        canonical_prefill_token_ids
    )

    first = api.serialize_decode_fixture(source)
    repeated = api.serialize_decode_fixture(source)

    assert first == repeated
    fixture_path = tmp_path / "decoder-decode-tied-top-tokens.safetensors"
    fixture_path.write_bytes(first)
    with safe_open(fixture_path, framework="pt", device="cpu") as reader:
        metadata = reader.metadata()
    assert metadata is not None
    assert metadata["generated_tokens"] == f"{PREFILL_TOKEN},{DECODE_TOKEN}"
    fixture_tensors = load_file(fixture_path, device="cpu")
    assert int(fixture_tensors["decoder.decode.00.input_token_id"][0, 0].item()) == PREFILL_TOKEN
    assert fixture_tensors["decoder.decode.00.logits"].shape == (1, V)


def test_serialize_decode_fixture_accepts_boundary_tied_top_tokens_with_backend_selected_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    source, boundary = _make_boundary_tied_top_tokens_source(api)
    prefill_logits = source.captured.stage_tensors["decoder.prefill.logits.last"][0]
    decode_logits = source.captured.deep_tensors["decoder.decode.00.logits"][0, 0]
    assert source.captured.token_trace is not None
    assert source.captured.token_trace.steps[0].top_tokens == (
        (PREFILL_TOKEN, 8.75),
        (boundary["prefill_selected_tied_tokens"][0], boundary["prefill_tied_score"]),
        (boundary["prefill_selected_tied_tokens"][1], boundary["prefill_tied_score"]),
    )
    assert source.captured.token_trace.steps[1].top_tokens == (
        (DECODE_TOKEN, 8.5),
        (boundary["decode_selected_tied_tokens"][0], boundary["decode_tied_score"]),
        (boundary["decode_selected_tied_tokens"][1], boundary["decode_tied_score"]),
    )
    _patch_topk_with_boundary_selected_subset(
        api,
        monkeypatch,
        prefill_logits=prefill_logits,
        decode_logits=decode_logits,
        prefill_selected_tied_tokens=boundary["prefill_selected_tied_tokens"],
        decode_selected_tied_tokens=boundary["decode_selected_tied_tokens"],
    )
    patched_prefill_scores, patched_prefill_token_ids = api.torch.topk(prefill_logits, k=3)
    patched_decode_scores, patched_decode_token_ids = api.torch.topk(decode_logits, k=3)
    assert tuple(int(token_id) for token_id in patched_prefill_token_ids.tolist()) == (
        PREFILL_TOKEN,
        boundary["prefill_selected_tied_tokens"][1],
        boundary["prefill_selected_tied_tokens"][0],
    )
    assert tuple(int(token_id) for token_id in patched_decode_token_ids.tolist()) == (
        DECODE_TOKEN,
        boundary["decode_selected_tied_tokens"][1],
        boundary["decode_selected_tied_tokens"][0],
    )
    assert patched_prefill_scores.tolist()[1] == patched_prefill_scores.tolist()[2]
    assert patched_decode_scores.tolist()[1] == patched_decode_scores.tolist()[2]

    first = api.serialize_decode_fixture(source)
    repeated = api.serialize_decode_fixture(source)

    assert first == repeated
    fixture_path = tmp_path / "decoder-decode-boundary-tied-top-tokens.safetensors"
    fixture_path.write_bytes(first)
    with safe_open(fixture_path, framework="pt", device="cpu") as reader:
        metadata = reader.metadata()
    assert metadata is not None
    assert metadata["generated_tokens"] == f"{PREFILL_TOKEN},{DECODE_TOKEN}"
    fixture_tensors = load_file(fixture_path, device="cpu")
    assert int(fixture_tensors["decoder.decode.00.input_token_id"][0, 0].item()) == PREFILL_TOKEN
    assert fixture_tensors["decoder.decode.00.logits"].shape == (1, V)


def test_serialize_decode_fixture_rejects_tied_trace_with_nonmember_token_even_if_score_looks_valid() -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    source = _make_tied_top_tokens_source(api)
    trace = source.captured.token_trace
    assert trace is not None
    prefill_logits = source.captured.stage_tensors["decoder.prefill.logits.last"][0]
    actual_top_tokens = _top_tokens_from_logits(prefill_logits, count=3)
    nonmember_token = 109
    assert nonmember_token not in {token_id for token_id, _ in actual_top_tokens}
    corrupted_trace = replace(
        trace,
        steps=(
            replace(
                trace.steps[0],
                top_tokens=(
                    actual_top_tokens[0],
                    (nonmember_token, actual_top_tokens[1][1]),
                    actual_top_tokens[2],
                ),
            ),
            trace.steps[1],
        ),
    )
    corrupted = replace(
        source,
        captured=replace(source.captured, token_trace=corrupted_trace),
    )

    with pytest.raises(api.DecodeFixtureError) as error:
        api.serialize_decode_fixture(corrupted)

    assert error.value.code == "invalid_generation_trace"


def test_serialize_decode_fixture_rejects_boundary_tied_trace_with_excluded_same_score_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    source, boundary = _make_boundary_tied_top_tokens_source(api)
    trace = source.captured.token_trace
    assert trace is not None
    prefill_logits = source.captured.stage_tensors["decoder.prefill.logits.last"][0]
    decode_logits = source.captured.deep_tensors["decoder.decode.00.logits"][0, 0]
    _patch_topk_with_boundary_selected_subset(
        api,
        monkeypatch,
        prefill_logits=prefill_logits,
        decode_logits=decode_logits,
        prefill_selected_tied_tokens=boundary["prefill_selected_tied_tokens"],
        decode_selected_tied_tokens=boundary["decode_selected_tied_tokens"],
    )
    corrupted_trace = replace(
        trace,
        steps=(
            replace(
                trace.steps[0],
                top_tokens=(
                    (PREFILL_TOKEN, 8.75),
                    (
                        boundary["prefill_excluded_tied_token"],
                        boundary["prefill_tied_score"],
                    ),
                    (
                        boundary["prefill_selected_tied_tokens"][0],
                        boundary["prefill_tied_score"],
                    ),
                ),
            ),
            replace(
                trace.steps[1],
                top_tokens=(
                    (DECODE_TOKEN, 8.5),
                    (
                        boundary["decode_excluded_tied_token"],
                        boundary["decode_tied_score"],
                    ),
                    (
                        boundary["decode_selected_tied_tokens"][0],
                        boundary["decode_tied_score"],
                    ),
                ),
            ),
        ),
    )
    corrupted = replace(
        source,
        captured=replace(source.captured, token_trace=corrupted_trace),
    )

    with pytest.raises(api.DecodeFixtureError) as error:
        api.serialize_decode_fixture(corrupted)

    assert error.value.code == "invalid_generation_trace"


def test_serialize_decode_fixture_rejects_tied_trace_with_wrong_exact_score() -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    source = _make_tied_top_tokens_source(api)
    trace = source.captured.token_trace
    assert trace is not None
    decode_logits = source.captured.deep_tensors["decoder.decode.00.logits"][0, 0]
    actual_top_tokens = _top_tokens_from_logits(decode_logits, count=3)
    assert actual_top_tokens[1][1] == actual_top_tokens[2][1]
    corrupted_trace = replace(
        trace,
        steps=(
            trace.steps[0],
            replace(
                trace.steps[1],
                top_tokens=(
                    actual_top_tokens[0],
                    actual_top_tokens[1],
                    (actual_top_tokens[2][0], actual_top_tokens[2][1] - 0.125),
                ),
            ),
        ),
    )
    corrupted = replace(
        source,
        captured=replace(source.captured, token_trace=corrupted_trace),
    )

    with pytest.raises(api.DecodeFixtureError) as error:
        api.serialize_decode_fixture(corrupted)

    assert error.value.code == "invalid_generation_trace"


def test_serialize_decode_fixture_rejects_noncanonical_tied_trace_order() -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    source = _make_tied_top_tokens_source(api)
    trace = source.captured.token_trace
    assert trace is not None
    decode_top_tokens = trace.steps[1].top_tokens
    assert decode_top_tokens[1][1] == decode_top_tokens[2][1]
    assert decode_top_tokens[1][0] < decode_top_tokens[2][0]
    corrupted_trace = replace(
        trace,
        steps=(
            trace.steps[0],
            replace(
                trace.steps[1],
                top_tokens=(
                    decode_top_tokens[0],
                    decode_top_tokens[2],
                    decode_top_tokens[1],
                ),
            ),
        ),
    )
    corrupted = replace(
        source,
        captured=replace(source.captured, token_trace=corrupted_trace),
    )

    with pytest.raises(api.DecodeFixtureError) as error:
        api.serialize_decode_fixture(corrupted)

    assert error.value.code == "invalid_generation_trace"


def test_serialize_decode_fixture_has_exact_m6_inventory(tmp_path: Path) -> None:
    decode_fixture = importlib.import_module("pvlc_reference.decode_fixture")
    captured, provenance = synthetic_capture(seed=20260719)
    source = _make_source(
        decode_fixture,
        seed=20260719,
        captured=captured,
        provenance=provenance,
    )
    assert set(source.__dataclass_fields__) == {
        "captured",
        "provenance",
        "publication",
        "publication_lock_blake3",
    }
    assert source.publication.bundle_digest == OFFICIAL_BUNDLE_DIGEST
    assert source.publication.semantic_fingerprint == OFFICIAL_SEMANTIC_FINGERPRINT
    assert source.publication.artifact_path == OFFICIAL_ARTIFACT_PATH
    assert source.publication.trace_level is TraceLevel.L3
    assert source.publication.generated_tokens == (PREFILL_TOKEN, DECODE_TOKEN)
    assert source.publication.decoded_text == "JUL"
    assert source.publication.repeat_count == 2

    serialized = decode_fixture.serialize_decode_fixture(source)
    assert isinstance(serialized, bytes)
    fixture_path = tmp_path / "decoder-decode-official-v1.safetensors"
    fixture_path.write_bytes(serialized)

    prefix = "decoder.decode.00"
    expected_shapes = {
        f"{prefix}.kv.key.layer_token_major": (L, S + 1, KVH, D),
        f"{prefix}.kv.value.layer_token_major": (L, S + 1, KVH, D),
        f"{prefix}.attention_mask": (1, S + 1),
        f"{prefix}.cache_position": (1,),
        f"{prefix}.input_token_id": (1, 1),
        f"{prefix}.position_ids": (3, 1, 1),
        "decoder.mrope.delta": (1, 1),
        f"{prefix}.rope.cos.axis_major": (3, 1, D),
        f"{prefix}.rope.sin.axis_major": (3, 1, D),
        f"{prefix}.layer.00.input": (1, H),
        f"{prefix}.layer.00.norm1": (1, H),
        f"{prefix}.layer.00.q": (1, QH * D),
        f"{prefix}.layer.00.k": (1, KVH * D),
        f"{prefix}.layer.00.v": (1, KVH * D),
        f"{prefix}.layer.00.mrope.q.token_major": (1, QH, D),
        f"{prefix}.layer.00.mrope.k.token_major": (1, KVH, D),
        f"{prefix}.layer.00.attention.context.token_major": (1, QH, D),
        f"{prefix}.layer.00.attention.output": (1, H),
        f"{prefix}.layer.00.attention.residual": (1, H),
        f"{prefix}.layer.00.norm2": (1, H),
        f"{prefix}.layer.00.mlp.gate": (1, I),
        f"{prefix}.layer.00.mlp.up": (1, I),
        f"{prefix}.layer.00.mlp.activation": (1, I),
        f"{prefix}.layer.00.mlp.down": (1, H),
        **{
            f"{prefix}.layer.{layer_index:02d}.output": (1, H)
            for layer_index in range(L)
        },
        f"{prefix}.final_norm": (1, H),
        f"{prefix}.logits": (1, V),
    }
    i64_names = {
        f"{prefix}.attention_mask",
        f"{prefix}.cache_position",
        f"{prefix}.input_token_id",
        f"{prefix}.position_ids",
        "decoder.mrope.delta",
    }
    expected_metadata = {
        "bias": "false",
        "cache_layout": "layer_token_major",
        "cache_position": str(S),
        "cache_tokens": str(S + 1),
        "capture_repeat_count": str(source.publication.repeat_count),
        "capture_tool_version": "0.1.0",
        "case_id": source.publication.case_id,
        "decode_input_token": str(PREFILL_TOKEN),
        "decode_next_token": str(DECODE_TOKEN),
        "decode_step": "1",
        "decode_tokens": "1",
        "decoded_text": source.publication.decoded_text,
        "device": "mps",
        "dtype": "bfloat16",
        "fixture_schema": "pvlc.decoder_decode.official.v1",
        "generated_tokens": ",".join(
            str(token) for token in source.publication.generated_tokens
        ),
        "head_dim": str(D),
        "hidden_size": str(H),
        "intermediate_size": str(I),
        "key_value_heads": str(KVH),
        "layer0_weights_fixture_blake3": (
            "blake3:30eed2f7a4d9336f8b3429d7294065c45214c4425f7559e8e2059ebd54c89522"
        ),
        "layers": str(L),
        "model_id": "PaddlePaddle/PaddleOCR-VL-1.6",
        "model_lock_blake3": (
            "blake3:c2ffa46821fdda1a7e4cd6bad63cdcdc2f775b8fed250b0679cc7d5411577d10"
        ),
        "model_revision": "66317acc4c9fc17bd154591ce650735cd2855f3e",
        "mrope_sections": "16,24,24",
        "oracle": "TransformersOracle pinned remote code",
        "prefix_tokens": str(S),
        "query_heads": str(QH),
        "rms_norm_epsilon": "1e-5",
        "rope_delta": "-290",
        "source_bundle_digest": source.publication.bundle_digest,
        "source_publication_lock_blake3": source.publication_lock_blake3,
        "source_semantic_fingerprint": source.publication.semantic_fingerprint,
        "torch_version": "2.13.0",
        "trace_level": source.publication.trace_level.value,
        "transformers_version": "5.14.1",
        "vocab_size": str(V),
    }
    assert len(expected_shapes) == 44
    assert len(expected_metadata) == 38

    with safe_open(fixture_path, framework="pt", device="cpu") as reader:
        assert set(reader.keys()) == set(expected_shapes)
        assert reader.metadata() == expected_metadata
    tensors = load_file(fixture_path, device="cpu")
    assert set(tensors) == set(expected_shapes)
    for semantic_id, expected_shape in expected_shapes.items():
        assert tuple(tensors[semantic_id].shape) == expected_shape
        expected_dtype = torch.int64 if semantic_id in i64_names else torch.bfloat16
        assert tensors[semantic_id].dtype == expected_dtype

    for layer_index in range(L):
        layer = f"{layer_index:02d}"
        for kind in ("key", "value"):
            stacked = tensors[f"{prefix}.kv.{kind}.layer_token_major"]
            prefill = captured.deep_tensors[f"decoder.layer.{layer}.kv.{kind}"]
            post = captured.deep_tensors[f"{prefix}.layer.{layer}.kv.{kind}"]
            expected_prefix = prefill[0].permute(1, 0, 2).contiguous()
            expected_append = post[0, :, S, :]
            assert torch.equal(stacked[layer_index, :S], expected_prefix)
            assert torch.equal(stacked[layer_index, S], expected_append)
        output_name = f"{prefix}.layer.{layer}.output"
        assert torch.equal(
            tensors[output_name], captured.deep_tensors[output_name].squeeze(0)
        )

    direct_layer0_names = (
        "input",
        "norm1",
        "q",
        "k",
        "v",
        "attention.output",
        "attention.residual",
        "norm2",
        "mlp.gate",
        "mlp.up",
        "mlp.activation",
        "mlp.down",
        "output",
    )
    for suffix in direct_layer0_names:
        output_name = f"{prefix}.layer.00.{suffix}"
        assert torch.equal(
            tensors[output_name], captured.deep_tensors[output_name].squeeze(0)
        )
    for suffix in ("mrope.q", "mrope.k"):
        source_name = f"{prefix}.layer.00.{suffix}"
        expected = (
            captured.deep_tensors[source_name]
            .squeeze(0)
            .permute(1, 0, 2)
            .contiguous()
        )
        assert torch.equal(tensors[f"{source_name}.token_major"], expected)
    context_name = f"{prefix}.layer.00.attention.context"
    assert torch.equal(
        tensors[f"{context_name}.token_major"],
        captured.deep_tensors[context_name].reshape(1, QH, D),
    )
    assert torch.equal(
        tensors[f"{prefix}.rope.cos.axis_major"],
        captured.deep_tensors[f"{prefix}.rope.cos"].squeeze(1),
    )
    assert torch.equal(
        tensors[f"{prefix}.rope.sin.axis_major"],
        captured.deep_tensors[f"{prefix}.rope.sin"].squeeze(1),
    )
    assert torch.equal(
        tensors[f"{prefix}.final_norm"],
        captured.deep_tensors[f"{prefix}.final_norm"].squeeze(0),
    )
    assert torch.equal(
        tensors[f"{prefix}.logits"],
        captured.deep_tensors[f"{prefix}.logits"].squeeze(0),
    )
    assert torch.equal(
        tensors[f"{prefix}.attention_mask"],
        captured.deep_tensors[f"{prefix}.attention_mask"],
    )
    assert torch.equal(
        tensors[f"{prefix}.cache_position"],
        captured.deep_tensors[f"{prefix}.cache_position"],
    )
    assert tensors[f"{prefix}.input_token_id"].tolist() == [[PREFILL_TOKEN]]
    assert torch.equal(
        tensors[f"{prefix}.position_ids"],
        captured.deep_tensors[f"{prefix}.position_ids"],
    )
    assert torch.equal(
        tensors["decoder.mrope.delta"],
        captured.stage_tensors["decoder.mrope.delta"],
    )
    assert tensors[f"{prefix}.attention_mask"].tolist() == [[1] * (S + 1)]
    assert tensors[f"{prefix}.cache_position"].tolist() == [S]
    assert tensors[f"{prefix}.position_ids"].tolist() == [[[42]], [[42]], [[42]]]
    assert tensors["decoder.mrope.delta"].tolist() == [[-290]]
    assert not any(name.endswith(".weight") for name in tensors)


def test_load_decode_fixture_source_verifies_and_round_trips_official_l3(
    verified_decode_source_bundles,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    bundle_root, _, publication, lock_path, lock_digest = (
        verified_decode_source_bundles["l3"]
    )
    before = _bundle_bytes_snapshot(bundle_root)
    lock_before = lock_path.read_bytes()

    loaded = api.load_decode_fixture_source(
        bundle_root,
        publication_lock_path=lock_path,
        expected_publication_lock_blake3=lock_digest,
    )

    assert _bundle_bytes_snapshot(bundle_root) == before
    assert lock_path.read_bytes() == lock_before
    assert loaded.publication == publication
    assert loaded.publication_lock_blake3 == lock_digest
    assert loaded.provenance == verified_decode_source_bundles["provenance"]
    assert loaded.captured.token_trace == verified_decode_source_bundles[
        "captured"
    ].token_trace
    _assert_captured_tensors_exact(
        loaded.captured, verified_decode_source_bundles["captured"]
    )

    direct = _make_source(
        api,
        seed=20260719,
        captured=verified_decode_source_bundles["captured"],
        provenance=verified_decode_source_bundles["provenance"],
        publication=publication,
        publication_lock_blake3=lock_digest,
    )
    assert api.serialize_decode_fixture(loaded) == api.serialize_decode_fixture(direct)


def test_load_decode_fixture_source_preserves_lower_ranked_top_tokens(
    verified_decode_source_bundles,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    bundle_root, _, publication, lock_path, lock_digest = (
        verified_decode_source_bundles["lower_ranked"]
    )
    expected_capture = verified_decode_source_bundles["lower_ranked_capture"]

    loaded = api.load_decode_fixture_source(
        bundle_root,
        publication_lock_path=lock_path,
        expected_publication_lock_blake3=lock_digest,
    )

    assert loaded.publication == publication
    assert loaded.captured.token_trace == expected_capture.token_trace
    assert loaded.captured.token_trace is not None
    assert loaded.captured.token_trace.steps[0].top_tokens == (
        (PREFILL_TOKEN, 8.75),
        (12345, 6.25),
        (77, -0.5),
    )
    assert loaded.captured.token_trace.steps[1].top_tokens == (
        (DECODE_TOKEN, 8.5),
        (23456, 7.0),
        (88, -1.25),
    )
    for logits, step in (
        (
            loaded.captured.stage_tensors["decoder.prefill.logits.last"][0],
            loaded.captured.token_trace.steps[0],
        ),
        (
            loaded.captured.deep_tensors["decoder.decode.00.logits"][0, 0],
            loaded.captured.token_trace.steps[1],
        ),
    ):
        assert _top_tokens_from_logits(logits, count=3) == step.top_tokens
    _assert_captured_tensors_exact(loaded.captured, expected_capture)


def test_load_decode_fixture_source_rejects_schema_invalid_verified_token_trace(
    verified_decode_source_bundles,
    tmp_path: Path,
) -> None:
    bundle_root, _, publication, _, _ = verified_decode_source_bundles["l3"]
    corrupted_root = tmp_path / "schema-invalid-token-trace-l3"
    shutil.copytree(bundle_root, corrupted_root)
    trace_path = corrupted_root / "token-trace.jsonl"
    records = [json.loads(line) for line in trace_path.read_bytes().splitlines()]
    del records[0]["top_tokens"]
    trace_path.write_bytes(b"".join(canonical_json_bytes(record) for record in records))
    rebuilt_bundle_digest = _rebuild_bundle_hashes(corrupted_root)
    report = verify_bundle(
        corrupted_root, expected_bundle_digest=rebuilt_bundle_digest
    )
    assert report.bundle_digest == rebuilt_bundle_digest

    corrupted_publication = replace(
        publication,
        artifact_path=corrupted_root.name,
        bundle_digest=rebuilt_bundle_digest,
    )
    lock_path = tmp_path / "schema-invalid-token-trace.golden.lock"
    lock_bytes = _publication_lock(corrupted_publication).canonical_bytes()
    lock_path.write_bytes(lock_bytes)
    lock_digest = f"blake3:{blake3(lock_bytes).hexdigest()}"

    api = importlib.import_module("pvlc_reference.decode_fixture")
    with pytest.raises(api.DecodeFixtureError) as error:
        api.load_decode_fixture_source(
            corrupted_root,
            publication_lock_path=lock_path,
            expected_publication_lock_blake3=lock_digest,
        )

    assert error.value.code == "invalid_generation_trace"


@pytest.mark.parametrize(
    "missing_pin",
    ("publication_lock_path", "expected_publication_lock_blake3"),
)
def test_load_decode_fixture_source_requires_all_publication_pins(
    verified_decode_source_bundles,
    missing_pin: str,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    bundle_root, _, _, lock_path, lock_digest = verified_decode_source_bundles["l3"]
    kwargs = {
        "publication_lock_path": lock_path,
        "expected_publication_lock_blake3": lock_digest,
    }
    del kwargs[missing_pin]

    with pytest.raises(TypeError):
        api.load_decode_fixture_source(bundle_root, **kwargs)


def test_load_decode_fixture_source_rejects_changed_fingerprint_under_old_lock_pin(
    verified_decode_source_bundles,
    tmp_path: Path,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    bundle_root, _, publication, _, lock_digest = verified_decode_source_bundles["l3"]
    changed_publication = replace(
        publication, semantic_fingerprint=f"blake3:{'f' * 64}"
    )
    changed_lock_path = tmp_path / "changed-fingerprint.golden.lock"
    changed_lock_path.write_bytes(_publication_lock(changed_publication).canonical_bytes())

    with pytest.raises(api.DecodeFixtureError) as error:
        api.load_decode_fixture_source(
            bundle_root,
            publication_lock_path=changed_lock_path,
            expected_publication_lock_blake3=lock_digest,
        )

    assert error.value.code == "publication_lock_digest_mismatch"


@pytest.mark.parametrize(
    "trailing_bytes",
    (b"\n", b" \t\n"),
    ids=("extra-newline", "extra-whitespace"),
)
def test_load_decode_fixture_source_rejects_exactly_hashed_noncanonical_lock_bytes(
    verified_decode_source_bundles,
    tmp_path: Path,
    trailing_bytes: bytes,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    bundle_root, _, publication, _, _ = verified_decode_source_bundles["l3"]
    local_bundle = tmp_path / publication.artifact_path
    shutil.copytree(bundle_root, local_bundle)
    canonical_lock = _publication_lock(publication)
    canonical_bytes = canonical_lock.canonical_bytes()
    raw_bytes = canonical_bytes + trailing_bytes
    lock_path = tmp_path / "noncanonical.golden.lock"
    lock_path.write_bytes(raw_bytes)
    exact_raw_digest = f"blake3:{blake3(raw_bytes).hexdigest()}"
    assert raw_bytes != canonical_bytes
    assert GoldenLock.load(lock_path) == canonical_lock.canonicalized()

    with pytest.raises(api.DecodeFixtureError) as error:
        api.load_decode_fixture_source(
            local_bundle,
            publication_lock_path=lock_path,
            expected_publication_lock_blake3=exact_raw_digest,
        )

    assert error.value.code == "invalid_source_publication"


def test_load_decode_fixture_source_rejects_exactly_hashed_canonical_multi_entry_lock(
    verified_decode_source_bundles,
    tmp_path: Path,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    bundle_root, _, publication, _, _ = verified_decode_source_bundles["l3"]
    local_bundle = tmp_path / publication.artifact_path
    shutil.copytree(bundle_root, local_bundle)
    other_publication = replace(
        publication,
        case_id="ocr.distinct_valid.0002",
        artifact_path="distinct-valid-l3",
        bundle_digest=f"blake3:{'a' * 64}",
        semantic_fingerprint=f"blake3:{'b' * 64}",
        generated_tokens=(101, 202),
        decoded_text="OTHER",
    )
    multi_entry_lock = GoldenLock(
        format_version=1,
        model_revision="66317acc4c9fc17bd154591ce650735cd2855f3e",
        trace_schema_version=1,
        bundles=(publication, other_publication),
    )
    lock_bytes = multi_entry_lock.canonical_bytes()
    lock_path = tmp_path / "multi-entry.golden.lock"
    lock_path.write_bytes(lock_bytes)
    exact_lock_digest = f"blake3:{blake3(lock_bytes).hexdigest()}"
    assert GoldenLock.load(lock_path) == multi_entry_lock.canonicalized()
    assert len(multi_entry_lock.bundles) == 2

    with pytest.raises(api.DecodeFixtureError) as error:
        api.load_decode_fixture_source(
            local_bundle,
            publication_lock_path=lock_path,
            expected_publication_lock_blake3=exact_lock_digest,
        )

    assert error.value.code == "invalid_source_publication"


@pytest.mark.parametrize(
    ("expected_lock_digest", "expected_code"),
    (
        ("blake3:short", "invalid_source_digest"),
        (f"blake3:{'e' * 64}", "publication_lock_digest_mismatch"),
    ),
)
def test_load_decode_fixture_source_rejects_bad_external_lock_pin(
    verified_decode_source_bundles,
    expected_lock_digest: str,
    expected_code: str,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    bundle_root, _, _, lock_path, _ = verified_decode_source_bundles["l3"]

    with pytest.raises(api.DecodeFixtureError) as error:
        api.load_decode_fixture_source(
            bundle_root,
            publication_lock_path=lock_path,
            expected_publication_lock_blake3=expected_lock_digest,
        )

    assert error.value.code == expected_code


def test_load_decode_fixture_source_rejects_bundle_not_selected_by_lock(
    verified_decode_source_bundles,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    bundle_root, _, _, _, _ = verified_decode_source_bundles["l3"]
    _, _, _, other_lock_path, other_lock_digest = verified_decode_source_bundles[
        "wrong_device"
    ]

    with pytest.raises(api.DecodeFixtureError) as error:
        api.load_decode_fixture_source(
            bundle_root,
            publication_lock_path=other_lock_path,
            expected_publication_lock_blake3=other_lock_digest,
        )

    assert error.value.code == "invalid_source_bundle"


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    (
        ("repeat_count", 1, "insufficient_repeat_count"),
        ("decoded_text", "JUX", "invalid_source_publication"),
    ),
)
def test_load_decode_fixture_source_rejects_wrong_locked_publication(
    verified_decode_source_bundles,
    tmp_path: Path,
    field: str,
    value: int | str,
    expected_code: str,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    bundle_root, _, publication, _, _ = verified_decode_source_bundles["l3"]
    local_bundle = tmp_path / publication.artifact_path
    local_bundle.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(bundle_root, local_bundle)
    lock_path = tmp_path / f"wrong-{field}.golden.lock"
    if field == "repeat_count":
        valid_bytes = _publication_lock(publication).canonical_bytes()
        lock_bytes = valid_bytes.replace(b"repeat_count = 2\n", b"repeat_count = 1\n")
    else:
        changed = replace(publication, decoded_text=value)
        lock_bytes = _publication_lock(changed).canonical_bytes()
    lock_path.write_bytes(lock_bytes)
    lock_digest = f"blake3:{blake3(lock_bytes).hexdigest()}"

    with pytest.raises(api.DecodeFixtureError) as error:
        api.load_decode_fixture_source(
            local_bundle,
            publication_lock_path=lock_path,
            expected_publication_lock_blake3=lock_digest,
        )

    assert error.value.code == expected_code


def test_load_decode_fixture_source_maps_artifact_integrity_failure(
    verified_decode_source_bundles,
    tmp_path: Path,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    bundle_root, _, publication, _, _ = verified_decode_source_bundles["l3"]
    corrupted_root = tmp_path / "corrupted-l3"
    shutil.copytree(bundle_root, corrupted_root)
    artifact_path = corrupted_root / "deep-checkpoints.safetensors"
    payload = bytearray(artifact_path.read_bytes())
    payload[-1] ^= 1
    artifact_path.write_bytes(payload)
    corrupted_publication = replace(
        publication, artifact_path=corrupted_root.name
    )
    lock_path = tmp_path / "corrupted.golden.lock"
    lock_bytes = _publication_lock(corrupted_publication).canonical_bytes()
    lock_path.write_bytes(lock_bytes)
    lock_digest = f"blake3:{blake3(lock_bytes).hexdigest()}"

    with pytest.raises(api.DecodeFixtureError) as error:
        api.load_decode_fixture_source(
            corrupted_root,
            publication_lock_path=lock_path,
            expected_publication_lock_blake3=lock_digest,
        )

    assert error.value.code == "invalid_source_bundle"


@pytest.mark.parametrize(
    ("artifact_name", "semantic_id"),
    (
        ("processor.safetensors", "processor.pixel_values"),
        ("stage-checkpoints.safetensors", "decoder.embedding"),
        ("deep-checkpoints.safetensors", "decoder.decode.00.final_norm"),
        ("token-trace.jsonl", None),
    ),
    ids=("processor", "stage", "deep", "token-trace"),
)
def test_load_decode_fixture_source_rejects_artifact_changed_after_verification(
    verified_decode_source_bundles,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
    semantic_id: str | None,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    bundle_root, _, publication, _, _ = verified_decode_source_bundles["l3"]
    local_bundle = tmp_path / publication.artifact_path
    shutil.copytree(bundle_root, local_bundle)

    local_publication = replace(publication, artifact_path=local_bundle.name)
    lock_path = tmp_path / "golden.lock"
    lock_bytes = _publication_lock(local_publication).canonical_bytes()
    lock_path.write_bytes(lock_bytes)
    lock_digest = f"blake3:{blake3(lock_bytes).hexdigest()}"

    original_verify_bundle = api.verify_bundle
    mutation_count = 0

    def verify_then_mutate(*args, **kwargs):
        nonlocal mutation_count
        report = original_verify_bundle(*args, **kwargs)
        if mutation_count == 0:
            artifact_path = local_bundle / artifact_name
            if semantic_id is None:
                records = [
                    json.loads(line) for line in artifact_path.read_bytes().splitlines()
                ]
                del records[0]["top_tokens"]
                artifact_path.write_bytes(
                    b"".join(canonical_json_bytes(record) for record in records)
                )
            else:
                _rewrite_safetensors_value(artifact_path, semantic_id)
            mutation_count += 1
        return report

    monkeypatch.setattr(api, "verify_bundle", verify_then_mutate)
    loaded_source = None

    with pytest.raises(api.DecodeFixtureError) as error:
        loaded_source = api.load_decode_fixture_source(
            local_bundle,
            publication_lock_path=lock_path,
            expected_publication_lock_blake3=lock_digest,
        )

    assert mutation_count == 1
    assert loaded_source is None
    assert error.value.code == "invalid_source_bundle"


@pytest.mark.parametrize(
    "artifact_name",
    POST_VERIFY_AUTHENTICATED_ARTIFACTS,
)
def test_load_decode_fixture_source_rejects_exact_byte_symlink_swap_after_verification(
    verified_decode_source_bundles,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    bundle_root, _, publication, _, _ = verified_decode_source_bundles["l3"]
    local_bundle, _, lock_path, lock_digest = _copy_locked_bundle(
        tmp_path,
        bundle_root,
        publication,
    )
    original_verify_bundle = api.verify_bundle
    mutation_count = 0

    def verify_then_swap(*args, **kwargs):
        nonlocal mutation_count
        report = original_verify_bundle(*args, **kwargs)
        if mutation_count == 0:
            artifact_path = local_bundle / artifact_name
            canary_path = tmp_path / f"{artifact_name}.canary"
            canary_path.parent.mkdir(parents=True, exist_ok=True)
            canary_path.write_bytes(artifact_path.read_bytes())
            artifact_path.unlink()
            artifact_path.symlink_to(canary_path)
            assert artifact_path.is_symlink()
            mutation_count += 1
        return report

    monkeypatch.setattr(api, "verify_bundle", verify_then_swap)
    loaded_source = None

    with pytest.raises(api.DecodeFixtureError) as error:
        loaded_source = api.load_decode_fixture_source(
            local_bundle,
            publication_lock_path=lock_path,
            expected_publication_lock_blake3=lock_digest,
        )

    assert mutation_count == 1
    assert loaded_source is None
    assert error.value.code == "invalid_source_bundle"


@pytest.mark.parametrize(
    "artifact_name",
    ("hashes.json", "token-trace.jsonl"),
)
def test_load_decode_fixture_source_rejects_fifo_swap_after_verification(
    verified_decode_source_bundles,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO guard requires mkfifo and SIGALRM support")

    api = importlib.import_module("pvlc_reference.decode_fixture")
    bundle_root, _, publication, _, _ = verified_decode_source_bundles["l3"]
    local_bundle, _, lock_path, lock_digest = _copy_locked_bundle(
        tmp_path,
        bundle_root,
        publication,
    )
    original_verify_bundle = api.verify_bundle
    mutation_count = 0

    def verify_then_swap(*args, **kwargs):
        nonlocal mutation_count
        report = original_verify_bundle(*args, **kwargs)
        if mutation_count == 0:
            artifact_path = local_bundle / artifact_name
            artifact_path.unlink()
            os.mkfifo(artifact_path, 0o600)
            mutation_count += 1
        return report

    monkeypatch.setattr(api, "verify_bundle", verify_then_swap)
    with _alarm_guard(artifact_name) as guard:
        with pytest.raises(api.DecodeFixtureError) as error:
            api.load_decode_fixture_source(
                local_bundle,
                publication_lock_path=lock_path,
                expected_publication_lock_blake3=lock_digest,
            )

    assert mutation_count == 1
    assert error.value.code == "invalid_source_bundle"
    assert not guard["fired"]


def test_load_decode_fixture_source_uses_held_lock_descriptor_after_path_swap(
    verified_decode_source_bundles,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo is required for lock path swap coverage")

    api = importlib.import_module("pvlc_reference.decode_fixture")
    bundle_root, _, publication, _, _ = verified_decode_source_bundles["l3"]
    local_bundle, local_publication, lock_path, lock_digest = _copy_locked_bundle(
        tmp_path,
        bundle_root,
        publication,
    )
    expected_lock_path = Path(os.path.abspath(lock_path))
    real_open = os.open
    mutation_count = 0
    lock_open_count = 0
    backup_path = tmp_path / "golden.lock.backup"

    def observe_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal mutation_count
        nonlocal lock_open_count
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        resolved = _resolve_open_target(path, dir_fd=dir_fd)
        if resolved == expected_lock_path:
            lock_open_count += 1
            if mutation_count == 0:
                _assert_open_flags_include_security_bits(flags)
                descriptor_stat = os.fstat(descriptor)
                assert stat.S_ISREG(descriptor_stat.st_mode)
                lock_path.rename(backup_path)
                os.mkfifo(lock_path, 0o600)
                mutation_count += 1
        return descriptor

    monkeypatch.setattr(api.os, "open", observe_open)
    with _alarm_guard("publication-lock") as guard:
        loaded = api.load_decode_fixture_source(
            local_bundle,
            publication_lock_path=lock_path,
            expected_publication_lock_blake3=lock_digest,
        )

    assert mutation_count == 1
    assert lock_open_count == 1
    assert not guard["fired"]
    assert loaded.publication == local_publication
    assert loaded.publication_lock_blake3 == lock_digest
    assert loaded.provenance == verified_decode_source_bundles["provenance"]
    _assert_captured_tensors_exact(
        loaded.captured,
        verified_decode_source_bundles["captured"],
    )


@pytest.mark.parametrize(
    "artifact_name",
    POST_VERIFY_AUTHENTICATED_ARTIFACTS,
)
def test_load_decode_fixture_source_uses_held_authenticated_payload_descriptor_after_verify(
    verified_decode_source_bundles,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo is required for payload path swap coverage")

    api = importlib.import_module("pvlc_reference.decode_fixture")
    bundle_root, _, publication, _, _ = verified_decode_source_bundles["l3"]
    local_bundle, local_publication, lock_path, lock_digest = _copy_locked_bundle(
        tmp_path,
        bundle_root,
        publication,
    )
    artifact_path = local_bundle / artifact_name
    backup_path = tmp_path / f"{artifact_name}.backup"
    original_verify_bundle = api.verify_bundle
    real_open = os.open
    verified = False
    mutation_count = 0
    open_state, observe_artifact_open = _make_bundle_relative_open_observer(
        local_bundle,
        artifact_name,
    )

    def verify_then_mark(*args, **kwargs):
        nonlocal verified
        report = original_verify_bundle(*args, **kwargs)
        verified = True
        return report

    def observe_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal mutation_count
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if observe_artifact_open(
            path,
            descriptor,
            dir_fd=dir_fd,
            verified=verified,
        ):
            if mutation_count == 0:
                _assert_open_flags_include_security_bits(flags)
                descriptor_stat = os.fstat(descriptor)
                assert stat.S_ISREG(descriptor_stat.st_mode)
                artifact_path.rename(backup_path)
                os.mkfifo(artifact_path, 0o600)
                mutation_count += 1
        return descriptor

    monkeypatch.setattr(api, "verify_bundle", verify_then_mark)
    monkeypatch.setattr(api.os, "open", observe_open)
    with _alarm_guard(artifact_name) as guard:
        loaded = api.load_decode_fixture_source(
            local_bundle,
            publication_lock_path=lock_path,
            expected_publication_lock_blake3=lock_digest,
        )

    assert mutation_count == 1
    assert open_state["bundle_dir_fd"] is not None
    assert open_state["bundle_dir_open_count"] == 1
    assert open_state["artifact_open_count"] == 1
    assert not guard["fired"]
    assert loaded.publication == local_publication
    assert loaded.publication_lock_blake3 == lock_digest
    assert loaded.provenance == verified_decode_source_bundles["provenance"]
    _assert_captured_tensors_exact(
        loaded.captured,
        verified_decode_source_bundles["captured"],
    )


@pytest.mark.parametrize(
    ("artifact_name", "canary_bytes"),
    (
        ("manifest.json", b"{\"bad\":true}"),
        ("processor.safetensors", b"not-safetensors"),
    ),
)
def test_load_decode_fixture_source_uses_held_authenticated_payload_descriptor_after_distinct_regular_swap(
    verified_decode_source_bundles,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
    canary_bytes: bytes,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    bundle_root, _, publication, _, _ = verified_decode_source_bundles["l3"]
    local_bundle, local_publication, lock_path, lock_digest = _copy_locked_bundle(
        tmp_path,
        bundle_root,
        publication,
    )
    artifact_path = local_bundle / artifact_name
    backup_path = tmp_path / f"{artifact_name}.regular-backup"
    original_bytes = artifact_path.read_bytes()
    assert original_bytes != canary_bytes
    original_verify_bundle = api.verify_bundle
    real_open = os.open
    verified = False
    mutation_count = 0
    open_state, observe_artifact_open = _make_bundle_relative_open_observer(
        local_bundle,
        artifact_name,
    )

    def verify_then_mark(*args, **kwargs):
        nonlocal verified
        report = original_verify_bundle(*args, **kwargs)
        verified = True
        return report

    def observe_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal mutation_count
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if observe_artifact_open(
            path,
            descriptor,
            dir_fd=dir_fd,
            verified=verified,
        ):
            if mutation_count == 0:
                _assert_open_flags_include_security_bits(flags)
                descriptor_stat = os.fstat(descriptor)
                assert stat.S_ISREG(descriptor_stat.st_mode)
                artifact_path.rename(backup_path)
                artifact_path.write_bytes(canary_bytes)
                assert artifact_path.read_bytes() == canary_bytes
                mutation_count += 1
        return descriptor

    monkeypatch.setattr(api, "verify_bundle", verify_then_mark)
    monkeypatch.setattr(api.os, "open", observe_open)
    with _alarm_guard(f"{artifact_name}-regular-canary") as guard:
        loaded = api.load_decode_fixture_source(
            local_bundle,
            publication_lock_path=lock_path,
            expected_publication_lock_blake3=lock_digest,
        )

    assert mutation_count == 1
    assert open_state["bundle_dir_fd"] is not None
    assert open_state["bundle_dir_open_count"] == 1
    assert open_state["artifact_open_count"] == 1
    assert not guard["fired"]
    assert loaded.publication == local_publication
    assert loaded.publication_lock_blake3 == lock_digest
    assert loaded.provenance == verified_decode_source_bundles["provenance"]
    _assert_captured_tensors_exact(
        loaded.captured,
        verified_decode_source_bundles["captured"],
    )


def test_load_decode_fixture_source_rejects_verified_l2_bundle(
    verified_decode_source_bundles,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    bundle_root, _, _, lock_path, lock_digest = verified_decode_source_bundles["l2"]

    with pytest.raises(api.DecodeFixtureError) as error:
        api.load_decode_fixture_source(
            bundle_root,
            publication_lock_path=lock_path,
            expected_publication_lock_blake3=lock_digest,
        )

    assert error.value.code == "invalid_source_bundle"


def test_load_decode_fixture_source_rejects_other_verified_case(
    verified_decode_source_bundles,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    bundle_root, _, _, lock_path, lock_digest = verified_decode_source_bundles[
        "wrong_case"
    ]

    with pytest.raises(api.DecodeFixtureError) as error:
        api.load_decode_fixture_source(
            bundle_root,
            publication_lock_path=lock_path,
            expected_publication_lock_blake3=lock_digest,
        )

    assert error.value.code == "invalid_source_bundle"


@pytest.mark.parametrize("profile", ("wrong_device", "wrong_dtype"))
def test_load_decode_fixture_source_rejects_other_verified_provenance(
    verified_decode_source_bundles,
    profile: str,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    bundle_root, _, _, lock_path, lock_digest = verified_decode_source_bundles[profile]

    with pytest.raises(api.DecodeFixtureError) as error:
        api.load_decode_fixture_source(
            bundle_root,
            publication_lock_path=lock_path,
            expected_publication_lock_blake3=lock_digest,
        )

    assert error.value.code == "invalid_provenance"


def test_export_decode_fixture_atomically_publishes_exact_bytes(
    tmp_path: Path,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    source = _make_source(api, seed=20260719)
    tensor_snapshots = _snapshot_source_tensors(source.captured)
    expected = api.serialize_decode_fixture(source)
    first_path = tmp_path / "one" / "nested" / "decode.safetensors"
    second_path = tmp_path / "two" / "decode.safetensors"

    api.export_decode_fixture(source, first_path)
    api.export_decode_fixture(source, second_path)

    assert first_path.read_bytes() == expected
    assert second_path.read_bytes() == expected
    assert first_path.read_bytes() == second_path.read_bytes()
    for path in (first_path, second_path):
        with safe_open(path, framework="pt", device="cpu") as reader:
            assert len(reader.keys()) == 44
        assert len(load_file(path, device="cpu")) == 44
    for tensor, snapshot in tensor_snapshots:
        assert tensor.dtype == snapshot.dtype
        assert tensor.shape == snapshot.shape
        assert torch.equal(tensor, snapshot)


def test_export_decode_fixture_fsyncs_staged_file_before_commit_and_parent_after(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    source = _make_source(api, seed=20260719)
    expected = api.serialize_decode_fixture(source)
    output_parent = tmp_path / "durable"
    output_parent.mkdir()
    output_path = output_parent / "decode.safetensors"
    original_fsync = api.os.fsync
    parent_descriptor = _open_directory(output_parent)
    parent_identity = _stat_identity(os.fstat(parent_descriptor))
    events: list[str] = []

    def observing_fsync(file_descriptor: int) -> None:
        descriptor_stat = os.fstat(file_descriptor)
        if (
            stat.S_ISREG(descriptor_stat.st_mode)
            and "staged_file_fsync" not in events
        ):
            assert _read_descriptor_bytes(file_descriptor) == expected
            matching_names = _entry_names_for_descriptor(
                parent_descriptor,
                file_descriptor,
            )
            output_visible = _entry_exists(
                output_path.name,
                directory_fd=parent_descriptor,
            )
            result = original_fsync(file_descriptor)
            assert matching_names == []
            assert not output_visible
            events.append("staged_file_fsync")
            return result

        is_output_parent = (
            stat.S_ISDIR(descriptor_stat.st_mode)
            and _stat_identity(descriptor_stat) == parent_identity
        )
        destination_visible = is_output_parent and _entry_exists(
            output_path.name,
            directory_fd=parent_descriptor,
        )
        if destination_visible:
            assert _read_bytes_at(
                output_path.name,
                directory_fd=parent_descriptor,
            ) == expected
        result = original_fsync(file_descriptor)
        if destination_visible:
            assert "descriptor_commit" in events
            events.append("parent_directory_fsync")
        return result

    def observing_commit(observation: _CommitObservation, commit):
        assert _read_commit_source_bytes(observation) == expected
        assert _commit_source_entry_names(observation) == []
        assert not _entry_exists(
            output_path.name,
            directory_fd=observation.parent_fd,
        )
        assert "staged_file_fsync" in events
        result = commit()
        assert _read_bytes_at(
            output_path.name,
            directory_fd=observation.parent_fd,
        ) == expected
        events.append("descriptor_commit")
        return result

    monkeypatch.setattr(api.os, "fsync", observing_fsync)
    _install_commit_wrapper(
        api,
        monkeypatch,
        output_parent=output_parent,
        output_name=output_path.name,
        parent_identity=parent_identity,
        handler=observing_commit,
    )
    try:
        api.export_decode_fixture(source, output_path)
    finally:
        os.close(parent_descriptor)

    commit_index = events.index("descriptor_commit")
    assert any(
        event == "staged_file_fsync"
        for event in events[:commit_index]
    )
    assert any(
        event == "parent_directory_fsync"
        for event in events[commit_index + 1 :]
    )
    assert output_path.read_bytes() == expected
    assert tuple(output_parent.iterdir()) == (output_path,)


def test_export_decode_fixture_anonymous_stage_never_links_staging_like_attacker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    source = _make_source(api, seed=20260719)
    expected = api.serialize_decode_fixture(source)
    attacker_payload = b"attacker-controlled-staging"
    assert len(attacker_payload) == 27

    output_parent = tmp_path / "anonymous-stage-attacker"
    output_parent.mkdir()
    output_path = output_parent / "decode.safetensors"
    parent_descriptor = _open_directory(output_parent)
    parent_identity = _stat_identity(os.fstat(parent_descriptor))
    original_fsync = api.os.fsync
    attacker_name = f".{output_path.name}.tmp-attacker"
    attacker_descriptors: list[int] = []
    attacker_identities: list[tuple[int, int]] = []
    attacker_nlinks: list[int] = []
    attacker_ctimes_ns: list[int] = []
    events: list[str] = []

    def install_staging_like_attacker_after_fsync(file_descriptor: int) -> None:
        descriptor_stat = os.fstat(file_descriptor)
        if attacker_descriptors or not stat.S_ISREG(descriptor_stat.st_mode):
            original_fsync(file_descriptor)
            return

        assert _read_descriptor_bytes(file_descriptor) == expected
        matching_names = _entry_names_for_descriptor(
            parent_descriptor,
            file_descriptor,
        )
        output_visible = _entry_exists(
            output_path.name,
            directory_fd=parent_descriptor,
        )
        original_fsync(file_descriptor)
        assert matching_names == []
        assert not output_visible

        attacker_descriptor = os.open(
            attacker_name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        try:
            _write_descriptor_all(attacker_descriptor, attacker_payload)
            original_fsync(attacker_descriptor)
            attacker_stat = os.fstat(attacker_descriptor)
        except Exception:
            os.close(attacker_descriptor)
            raise
        assert stat.S_ISREG(attacker_stat.st_mode)
        assert _read_bytes_at(
            attacker_name,
            directory_fd=parent_descriptor,
        ) == attacker_payload
        attacker_descriptors.append(attacker_descriptor)
        attacker_identities.append(_stat_identity(attacker_stat))
        attacker_nlinks.append(attacker_stat.st_nlink)
        attacker_ctimes_ns.append(attacker_stat.st_ctime_ns)
        events.append("staged_file_fsync")

    def observe_commit(observation: _CommitObservation, commit):
        assert events == ["staged_file_fsync"]
        assert len(attacker_descriptors) == 1
        assert _read_commit_source_bytes(observation) == expected
        assert _commit_source_entry_names(observation) == []
        result = commit()
        assert _read_bytes_at(
            output_path.name,
            directory_fd=observation.parent_fd,
        ) == expected
        attacker_stat = os.fstat(attacker_descriptors[0])
        attacker_entry_stat = os.stat(
            attacker_name,
            dir_fd=observation.parent_fd,
            follow_symlinks=False,
        )
        assert _stat_identity(attacker_stat) == attacker_identities[0]
        assert _stat_identity(attacker_entry_stat) == attacker_identities[0]
        assert attacker_stat.st_nlink == attacker_nlinks[0]
        assert attacker_entry_stat.st_nlink == attacker_nlinks[0]
        assert attacker_stat.st_ctime_ns == attacker_ctimes_ns[0]
        assert attacker_entry_stat.st_ctime_ns == attacker_ctimes_ns[0]
        events.append("descriptor_commit")
        return result

    monkeypatch.setattr(api.os, "fsync", install_staging_like_attacker_after_fsync)
    _install_commit_wrapper(
        api,
        monkeypatch,
        output_parent=output_parent,
        output_name=output_path.name,
        parent_identity=parent_identity,
        handler=observe_commit,
    )
    try:
        api.export_decode_fixture(source, output_path)

        assert events == ["staged_file_fsync", "descriptor_commit"]
        assert output_path.read_bytes() == expected
        assert len(attacker_descriptors) == 1
        assert _read_bytes_at(
            attacker_name,
            directory_fd=parent_descriptor,
        ) == attacker_payload
        assert sorted(os.listdir(parent_descriptor)) == sorted(
            (attacker_name, output_path.name)
        )
        attacker_stat = os.fstat(attacker_descriptors[0])
        attacker_entry_stat = os.stat(
            attacker_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        assert _stat_identity(attacker_stat) == attacker_identities[0]
        assert _stat_identity(attacker_entry_stat) == attacker_identities[0]
        assert attacker_stat.st_nlink == attacker_nlinks[0]
        assert attacker_entry_stat.st_nlink == attacker_nlinks[0]
        assert attacker_stat.st_ctime_ns == attacker_ctimes_ns[0]
        assert attacker_entry_stat.st_ctime_ns == attacker_ctimes_ns[0]
    finally:
        for attacker_descriptor in attacker_descriptors:
            os.close(attacker_descriptor)
        os.close(parent_descriptor)


@pytest.mark.skipif(
    sys.platform not in {"darwin", "linux"},
    reason="fd-bound publication adapter is required on Darwin and Linux",
)
@pytest.mark.parametrize(
    "payload",
    (
        pytest.param(b"\x00fd-bound-canary\xff\n", id="binary-short"),
        pytest.param(bytes(range(256)) * 17 + b"fd-bound-tail", id="multi-block"),
    ),
)
def test_fd_bound_commit_adapter_publishes_arbitrary_anonymous_payload_exclusively(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    create_anonymous_staging_descriptor = getattr(
        api,
        "_create_anonymous_staging_descriptor",
        None,
    )
    commit_staging_descriptor = getattr(
        api,
        "_commit_staging_descriptor",
        None,
    )
    assert callable(create_anonymous_staging_descriptor)
    assert callable(commit_staging_descriptor)
    primitive_name = (
        "_darwin_fclonefileat"
        if sys.platform == "darwin"
        else "_linux_linkat_anonymous"
    )
    native_commit_primitive = getattr(api, primitive_name, None)
    assert callable(native_commit_primitive)

    output_parent = tmp_path / "fd-bound-adapter"
    output_parent.mkdir()
    output_name = "canary.bin"
    parent_descriptor = _open_directory(output_parent)
    staging_descriptor: int | None = None
    retained_reader_descriptor: int | None = None
    try:
        staging_descriptor = create_anonymous_staging_descriptor(parent_descriptor)
        assert isinstance(staging_descriptor, int)
        staging_stat = os.fstat(staging_descriptor)
        assert stat.S_ISREG(staging_stat.st_mode)
        assert _entry_names_for_descriptor(
            parent_descriptor,
            staging_descriptor,
        ) == []
        assert not _entry_exists(output_name, directory_fd=parent_descriptor)

        _write_descriptor_all(staging_descriptor, payload)
        os.fsync(staging_descriptor)
        assert _read_descriptor_bytes(staging_descriptor) == payload
        assert _entry_names_for_descriptor(
            parent_descriptor,
            staging_descriptor,
        ) == []

        primitive_attempts: list[str] = []
        primitive_events: list[bytes] = []

        def observe_native_commit(
            source_fd: int,
            destination_parent_fd: int,
            destination_name: str,
        ):
            assert source_fd == staging_descriptor
            assert _stat_identity(
                os.fstat(destination_parent_fd)
            ) == _stat_identity(os.fstat(parent_descriptor))
            assert destination_name == output_name
            if primitive_events:
                primitive_attempts.append("repeat")
                assert primitive_events == [payload]
                assert _read_bytes_at(
                    destination_name,
                    directory_fd=destination_parent_fd,
                ) == payload
                return native_commit_primitive(
                    source_fd,
                    destination_parent_fd,
                    destination_name,
                )

            primitive_attempts.append("initial")
            assert stat.S_ISREG(os.fstat(source_fd).st_mode)
            assert _read_descriptor_bytes(source_fd) == payload
            assert _entry_names_for_descriptor(
                parent_descriptor,
                source_fd,
            ) == []
            assert not _entry_exists(
                destination_name,
                directory_fd=destination_parent_fd,
            )

            result = native_commit_primitive(
                source_fd,
                destination_parent_fd,
                destination_name,
            )

            committed_stat = os.stat(
                destination_name,
                dir_fd=destination_parent_fd,
                follow_symlinks=False,
            )
            assert stat.S_ISREG(committed_stat.st_mode)
            assert _read_bytes_at(
                destination_name,
                directory_fd=destination_parent_fd,
            ) == payload
            primitive_events.append(payload)
            return result

        monkeypatch.setattr(api, primitive_name, observe_native_commit)
        commit_staging_descriptor(
            staging_descriptor,
            parent_fd=parent_descriptor,
            output_name=output_name,
        )
        assert primitive_attempts == ["initial"]
        assert primitive_events == [payload]
        destination_stat = os.stat(
            output_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        assert stat.S_ISREG(destination_stat.st_mode)
        retained_reader_descriptor = os.open(
            output_name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        retained_reader_stat = os.fstat(retained_reader_descriptor)
        assert stat.S_ISREG(retained_reader_stat.st_mode)
        assert _stat_identity(retained_reader_stat) == _stat_identity(
            destination_stat
        )
        assert _read_descriptor_bytes(retained_reader_descriptor) == payload
        assert _read_bytes_at(
            output_name,
            directory_fd=parent_descriptor,
        ) == payload

        with pytest.raises(OSError) as error:
            commit_staging_descriptor(
                staging_descriptor,
                parent_fd=parent_descriptor,
                output_name=output_name,
            )

        assert error.value.errno == errno.EEXIST
        assert primitive_attempts == ["initial", "repeat"]
        assert primitive_events == [payload]
        repeated_destination_stat = os.stat(
            output_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        assert _stat_identity(repeated_destination_stat) == _stat_identity(
            destination_stat
        )
        assert _read_descriptor_bytes(retained_reader_descriptor) == payload
        assert _read_bytes_at(
            output_name,
            directory_fd=parent_descriptor,
        ) == payload
        assert os.listdir(parent_descriptor) == [output_name]
    finally:
        if retained_reader_descriptor is not None:
            os.close(retained_reader_descriptor)
        if staging_descriptor is not None:
            os.close(staging_descriptor)
        os.close(parent_descriptor)


@pytest.mark.skipif(
    sys.platform not in {"darwin", "linux"},
    reason="unsupported-platform dispatch is exercised from a supported host",
)
def test_fd_bound_commit_adapter_fails_closed_on_unsupported_platform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    create_anonymous_staging_descriptor = getattr(
        api,
        "_create_anonymous_staging_descriptor",
        None,
    )
    commit_staging_descriptor = getattr(
        api,
        "_commit_staging_descriptor",
        None,
    )
    darwin_commit_primitive = getattr(api, "_darwin_fclonefileat", None)
    linux_commit_primitive = getattr(api, "_linux_linkat_anonymous", None)
    assert callable(create_anonymous_staging_descriptor)
    assert callable(commit_staging_descriptor)
    assert callable(darwin_commit_primitive)
    assert callable(linux_commit_primitive)

    output_parent = tmp_path / "unsupported-fd-bound-adapter"
    output_parent.mkdir()
    output_name = "must-not-exist.bin"
    payload = b"unsupported-platform-canary"
    parent_descriptor = _open_directory(output_parent)
    staging_descriptor: int | None = None
    try:
        staging_descriptor = create_anonymous_staging_descriptor(parent_descriptor)
        assert isinstance(staging_descriptor, int)
        assert stat.S_ISREG(os.fstat(staging_descriptor).st_mode)
        assert _entry_names_for_descriptor(
            parent_descriptor,
            staging_descriptor,
        ) == []
        _write_descriptor_all(staging_descriptor, payload)
        os.fsync(staging_descriptor)

        primitive_calls: list[str] = []

        def reject_darwin_primitive(*args, **kwargs):
            primitive_calls.append("darwin")
            raise AssertionError("unsupported dispatch called Darwin primitive")

        def reject_linux_primitive(*args, **kwargs):
            primitive_calls.append("linux")
            raise AssertionError("unsupported dispatch called Linux primitive")

        monkeypatch.setattr(api, "_darwin_fclonefileat", reject_darwin_primitive)
        monkeypatch.setattr(api, "_linux_linkat_anonymous", reject_linux_primitive)
        monkeypatch.setattr(api.sys, "platform", "unsupported-test-platform")
        with pytest.raises(OSError) as error:
            commit_staging_descriptor(
                staging_descriptor,
                parent_fd=parent_descriptor,
                output_name=output_name,
            )

        unsupported_errnos = {
            errno.ENOTSUP,
            getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
        }
        assert error.value.errno in unsupported_errnos
        assert primitive_calls == []
        assert not _entry_exists(output_name, directory_fd=parent_descriptor)
        assert _entry_names_for_descriptor(
            parent_descriptor,
            staging_descriptor,
        ) == []
        assert _read_descriptor_bytes(staging_descriptor) == payload
        assert os.listdir(parent_descriptor) == []
    finally:
        if staging_descriptor is not None:
            os.close(staging_descriptor)
        os.close(parent_descriptor)


def test_export_decode_fixture_reviewer_race_never_publishes_attacker_inode_or_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    source = _make_source(api, seed=20260719)
    expected = api.serialize_decode_fixture(source)
    attacker_payload = b"attacker-link-race-payload"

    output_parent = tmp_path / "reviewer-race"
    output_parent.mkdir()
    output_path = output_parent / "decode.safetensors"
    parent_descriptor = _open_directory(output_parent)
    parent_identity = _stat_identity(os.fstat(parent_descriptor))
    uses_descriptor_commit = callable(
        getattr(api, "_commit_staging_descriptor", None)
    )
    staging_prechecked = False
    attacker_descriptors: list[int] = []
    retained_reader_descriptors: list[int] = []
    race_observations: list[dict[str, object]] = []

    if not uses_descriptor_commit:
        original_assert_owned_relative_entry = api._assert_owned_relative_entry

        def record_staging_precheck(*args, **kwargs) -> None:
            nonlocal staging_prechecked
            original_assert_owned_relative_entry(*args, **kwargs)
            if kwargs.get("label") == "staging entry":
                staging_prechecked = True

        monkeypatch.setattr(api, "_assert_owned_relative_entry", record_staging_precheck)

    def observe_commit(observation: _CommitObservation, commit):
        if observation.mode == "fd":
            assert _read_commit_source_bytes(observation) == expected
            assert _commit_source_entry_names(observation) == []
            result = commit()
            destination_stat = os.stat(
                observation.output_name,
                dir_fd=observation.parent_fd,
                follow_symlinks=False,
            )
            retained_reader_descriptor = os.open(
                observation.output_name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=observation.parent_fd,
            )
            retained_reader_descriptors.append(retained_reader_descriptor)
            race_observations.append(
                {
                    "mode": "fd",
                    "destination_stat": destination_stat,
                }
            )
            return result

        assert staging_prechecked
        assert observation.source_name is not None
        assert _read_commit_source_bytes(observation) == expected
        assert _commit_source_entry_names(observation) == [observation.source_name]

        os.unlink(observation.source_name, dir_fd=observation.parent_fd)
        attacker_descriptor = os.open(
            observation.source_name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=observation.parent_fd,
        )
        try:
            _write_descriptor_all(attacker_descriptor, attacker_payload)
            api.os.fsync(attacker_descriptor)
            attacker_pre_link_stat = os.fstat(attacker_descriptor)
            result = commit()
            attacker_post_link_stat = os.fstat(attacker_descriptor)
            destination_stat = os.stat(
                observation.output_name,
                dir_fd=observation.parent_fd,
                follow_symlinks=False,
            )
            retained_reader_descriptor = os.open(
                observation.output_name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=observation.parent_fd,
            )
        except Exception:
            os.close(attacker_descriptor)
            raise

        attacker_descriptors.append(attacker_descriptor)
        retained_reader_descriptors.append(retained_reader_descriptor)
        race_observations.append(
            {
                "mode": "link",
                "staged_name": observation.source_name,
                "attacker_pre_link_stat": attacker_pre_link_stat,
                "attacker_post_link_stat": attacker_post_link_stat,
                "destination_stat": destination_stat,
            }
        )
        return result

    commit_mode = _install_commit_wrapper(
        api,
        monkeypatch,
        output_parent=output_parent,
        output_name=output_path.name,
        parent_identity=parent_identity,
        handler=observe_commit,
    )
    export_error: api.DecodeFixtureError | None = None
    try:
        try:
            api.export_decode_fixture(source, output_path)
        except api.DecodeFixtureError as error:
            export_error = error

        if commit_mode == "link":
            assert export_error is not None
            assert export_error.code == "publication_failed"
            assert len(race_observations) == 1
            observation = race_observations[0]
            assert observation["mode"] == "link"
            attacker_pre_link_stat = observation["attacker_pre_link_stat"]
            attacker_post_link_stat = observation["attacker_post_link_stat"]
            destination_stat = observation["destination_stat"]
            staged_name = observation["staged_name"]
            assert isinstance(attacker_pre_link_stat, os.stat_result)
            assert isinstance(attacker_post_link_stat, os.stat_result)
            assert isinstance(destination_stat, os.stat_result)
            assert isinstance(staged_name, str)
            assert attacker_post_link_stat.st_nlink == attacker_pre_link_stat.st_nlink
            assert (
                attacker_post_link_stat.st_ctime_ns
                == attacker_pre_link_stat.st_ctime_ns
            )
            assert _stat_identity(destination_stat) != _stat_identity(
                attacker_pre_link_stat
            )
            assert _read_descriptor_bytes(retained_reader_descriptors[0]) != attacker_payload
            assert not _entry_exists(output_path.name, directory_fd=parent_descriptor)
            assert _entry_exists(staged_name, directory_fd=parent_descriptor)
            assert _read_bytes_at(staged_name, directory_fd=parent_descriptor) == attacker_payload
            return

        assert commit_mode == "fd"
        assert export_error is None
        assert len(race_observations) == 1
        observation = race_observations[0]
        assert observation["mode"] == "fd"
        destination_stat = observation["destination_stat"]
        assert isinstance(destination_stat, os.stat_result)
        assert stat.S_ISREG(destination_stat.st_mode)
        assert len(retained_reader_descriptors) == 1
        retained_reader_stat = os.fstat(retained_reader_descriptors[0])
        assert stat.S_ISREG(retained_reader_stat.st_mode)
        assert _stat_identity(retained_reader_stat) == _stat_identity(
            destination_stat
        )
        assert _read_descriptor_bytes(retained_reader_descriptors[0]) == expected
        assert _read_descriptor_bytes(retained_reader_descriptors[0]) != attacker_payload
        assert output_path.read_bytes() == expected
        assert output_path.read_bytes() != attacker_payload
        assert tuple(output_parent.iterdir()) == (output_path,)
    finally:
        for descriptor in retained_reader_descriptors:
            os.close(descriptor)
        for descriptor in attacker_descriptors:
            os.close(descriptor)
        os.close(parent_descriptor)


def test_export_decode_fixture_rejects_parent_replacement_at_commit_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    source = _make_source(api, seed=20260719)
    expected = api.serialize_decode_fixture(source)
    attacker_payload = b"attacker-controlled-new-parent"
    assert len(attacker_payload) == 30

    output_parent = tmp_path / "publication-parent"
    relocated_parent = tmp_path / "relocated-authentic-parent"
    output_parent.mkdir()
    output_path = output_parent / "decode.safetensors"
    parent_descriptor = _open_directory(output_parent)
    parent_identity = _stat_identity(os.fstat(parent_descriptor))
    observed_commits: list[tuple[str, str]] = []

    def replace_parent_then_commit(observation: _CommitObservation, commit):
        assert not observed_commits
        assert _read_commit_source_bytes(observation) == expected
        source_names = _commit_source_entry_names(observation)
        if observation.mode == "fd":
            assert source_names == []
            attacker_name = f".{output_path.name}.tmp-attacker-parent-swap"
        else:
            assert observation.source_name is not None
            assert source_names == [observation.source_name]
            attacker_name = observation.source_name

        output_parent.rename(relocated_parent)
        output_parent.mkdir()
        replacement_parent_descriptor = _open_directory(output_parent)
        try:
            _write_bytes_exclusive_at(
                attacker_name,
                attacker_payload,
                directory_fd=replacement_parent_descriptor,
            )
            assert _read_bytes_at(
                attacker_name,
                directory_fd=replacement_parent_descriptor,
            ) == attacker_payload
            assert _stat_identity(
                os.fstat(replacement_parent_descriptor)
            ) != parent_identity
        finally:
            os.close(replacement_parent_descriptor)

        assert _stat_identity(os.fstat(parent_descriptor)) == parent_identity
        assert _stat_identity(
            os.stat(relocated_parent, follow_symlinks=False)
        ) == parent_identity
        result = commit()
        assert _read_bytes_at(
            output_path.name,
            directory_fd=observation.parent_fd,
        ) == expected
        observed_commits.append((observation.mode, attacker_name))
        return result

    _install_commit_wrapper(
        api,
        monkeypatch,
        output_parent=output_parent,
        output_name=output_path.name,
        parent_identity=parent_identity,
        handler=replace_parent_then_commit,
    )
    try:
        with pytest.raises(api.DecodeFixtureError) as error:
            api.export_decode_fixture(source, output_path)

        assert error.value.code == "publication_failed"
        assert len(observed_commits) == 1
        commit_mode, attacker_name = observed_commits[0]
        assert commit_mode in {"fd", "link"}
        assert not _entry_exists(
            output_path.name,
            directory_fd=parent_descriptor,
        )
        assert os.listdir(parent_descriptor) == []

        replacement_parent_descriptor = _open_directory(output_parent)
        try:
            assert not _entry_exists(
                output_path.name,
                directory_fd=replacement_parent_descriptor,
            )
            assert _read_bytes_at(
                attacker_name,
                directory_fd=replacement_parent_descriptor,
            ) == attacker_payload
            assert os.listdir(replacement_parent_descriptor) == [attacker_name]
        finally:
            os.close(replacement_parent_descriptor)
        assert relocated_parent.is_dir()
        assert _stat_identity(
            os.stat(relocated_parent, follow_symlinks=False)
        ) == parent_identity
        assert not os.path.lexists(relocated_parent / output_path.name)
        assert tuple(relocated_parent.iterdir()) == ()
    finally:
        os.close(parent_descriptor)


def test_export_decode_fixture_commit_failure_cleans_staging_and_new_parent_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    source = _make_source(api, seed=20260719)
    expected = api.serialize_decode_fixture(source)
    output_path = tmp_path / "new" / "nested" / "decode.safetensors"
    observed_commits: list[str] = []

    def fail_commit(observation: _CommitObservation, _commit):
        assert _read_commit_source_bytes(observation) == expected
        source_names = _commit_source_entry_names(observation)
        if observation.mode == "fd":
            assert source_names == []
        else:
            assert observation.source_name is not None
            assert source_names == [observation.source_name]
        assert not _entry_exists(
            output_path.name,
            directory_fd=observation.parent_fd,
        )
        observed_commits.append(observation.mode)
        raise OSError(errno.EIO, "forced descriptor-commit publication failure")

    _install_commit_wrapper(
        api,
        monkeypatch,
        output_parent=output_path.parent,
        output_name=output_path.name,
        parent_identity=None,
        handler=fail_commit,
    )
    with pytest.raises(api.DecodeFixtureError) as error:
        api.export_decode_fixture(source, output_path)

    assert error.value.code == "publication_failed"
    assert len(observed_commits) == 1
    assert observed_commits[0] in {"fd", "link"}
    assert not os.path.lexists(output_path)
    assert tuple(tmp_path.rglob("*")) == ()


def test_export_decode_fixture_post_commit_destination_stat_failure_cleans_output_and_keeps_existing_parent_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    source = _make_source(api, seed=20260719)
    expected = api.serialize_decode_fixture(source)
    output_parent = tmp_path / "publish"
    output_parent.mkdir()
    output_path = output_parent / "decode.safetensors"
    parent_descriptor = _open_directory(output_parent)
    parent_identity = _stat_identity(os.fstat(parent_descriptor))
    original_stat_relative_entry = api._stat_relative_entry
    observed_non_destination_names: list[str] = []
    injected_failures = 0

    def fail_post_link_destination_stat(name: str, *, directory_fd: int) -> os.stat_result:
        nonlocal injected_failures
        assert _stat_identity(os.fstat(directory_fd)) == parent_identity

        if name != output_path.name:
            observed_non_destination_names.append(name)
            return original_stat_relative_entry(name, directory_fd=directory_fd)

        destination_stat = original_stat_relative_entry(name, directory_fd=directory_fd)
        if injected_failures:
            return destination_stat
        assert stat.S_ISREG(destination_stat.st_mode)
        assert _read_bytes_at(name, directory_fd=directory_fd) == expected
        injected_failures += 1
        raise OSError(
            errno.EIO,
            "simulated post-link destination stat failure",
        )

    monkeypatch.setattr(
        api,
        "_stat_relative_entry",
        fail_post_link_destination_stat,
    )
    try:
        with pytest.raises(api.DecodeFixtureError) as error:
            api.export_decode_fixture(source, output_path)

        assert error.value.code == "publication_failed"
        assert injected_failures == 1
        assert all(
            name.startswith(f".{output_path.name}.tmp-")
            for name in observed_non_destination_names
        )
        assert not _entry_exists(output_path.name, directory_fd=parent_descriptor)
        assert os.listdir(parent_descriptor) == []
        assert tuple(output_parent.iterdir()) == ()
    finally:
        os.close(parent_descriptor)


def test_export_decode_fixture_file_fsync_failure_cleans_staging_and_new_parent_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    source = _make_source(api, seed=20260719)
    expected = api.serialize_decode_fixture(source)
    output_path = tmp_path / "new" / "nested" / "decode.safetensors"
    original_fsync = api.os.fsync
    observed_staging: list[tuple[tuple[int, int], int]] = []

    def fail_staging_file_fsync(file_descriptor: int) -> None:
        descriptor_stat = os.fstat(file_descriptor)
        if not stat.S_ISREG(descriptor_stat.st_mode) or observed_staging:
            original_fsync(file_descriptor)
            return

        assert _read_descriptor_bytes(file_descriptor) == expected
        assert not os.path.lexists(output_path)
        observed_staging.append(
            (_stat_identity(descriptor_stat), descriptor_stat.st_size)
        )
        raise OSError(errno.EIO, "forced staging-file fsync failure")

    monkeypatch.setattr(api.os, "fsync", fail_staging_file_fsync)
    with pytest.raises(api.DecodeFixtureError) as error:
        api.export_decode_fixture(source, output_path)

    assert error.value.code == "publication_failed"
    assert len(observed_staging) == 1
    assert observed_staging[0][1] == len(expected)
    assert not os.path.lexists(output_path)
    assert tuple(tmp_path.rglob("*")) == ()


def test_export_decode_fixture_fsync_observes_anonymous_staging_until_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    source = _make_source(api, seed=20260719)
    expected = api.serialize_decode_fixture(source)
    output_parent = tmp_path / "anonymous-stage"
    output_parent.mkdir()
    output_path = output_parent / "decode.safetensors"
    original_fsync = api.os.fsync
    parent_descriptor = _open_directory(output_parent)
    parent_identity = _stat_identity(os.fstat(parent_descriptor))
    events: list[str] = []

    def observe_fsync(file_descriptor: int) -> None:
        descriptor_stat = os.fstat(file_descriptor)
        if (
            stat.S_ISREG(descriptor_stat.st_mode)
            and "payload_file_fsync" not in events
        ):
            assert _read_descriptor_bytes(file_descriptor) == expected
            assert _stat_identity(os.fstat(parent_descriptor)) == parent_identity
            matching_names = _entry_names_for_descriptor(
                parent_descriptor,
                file_descriptor,
            )
            output_visible = _entry_exists(
                output_path.name,
                directory_fd=parent_descriptor,
            )
            result = original_fsync(file_descriptor)
            assert matching_names == []
            assert not output_visible
            events.append("payload_file_fsync")
            return result

        destination_visible = (
            stat.S_ISDIR(descriptor_stat.st_mode)
            and _stat_identity(descriptor_stat) == parent_identity
            and _entry_exists(output_path.name, directory_fd=parent_descriptor)
        )
        if destination_visible:
            assert _read_bytes_at(output_path.name, directory_fd=parent_descriptor) == expected
        result = original_fsync(file_descriptor)
        if destination_visible:
            events.append("parent_directory_fsync")
        return result

    monkeypatch.setattr(api.os, "fsync", observe_fsync)
    try:
        api.export_decode_fixture(source, output_path)
    finally:
        os.close(parent_descriptor)

    assert events.index("payload_file_fsync") < events.index("parent_directory_fsync")
    assert output_path.read_bytes() == expected
    assert tuple(output_parent.iterdir()) == (output_path,)


def test_export_decode_fixture_commit_race_has_exactly_one_whole_payload_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    first_source = _make_source(api, seed=20260719)
    second_source = _make_source(api, seed=20260720)
    first_payload = api.serialize_decode_fixture(first_source)
    second_payload = api.serialize_decode_fixture(second_source)
    assert first_payload != second_payload

    output_parent = tmp_path / "race"
    output_parent.mkdir()
    output_path = output_parent / "decode.safetensors"
    parent_descriptor = _open_directory(output_parent)
    parent_identity = _stat_identity(os.fstat(parent_descriptor))
    commit_barrier = threading.Barrier(2)
    observation_lock = threading.Lock()
    observed_commits: list[tuple[str, tuple[str, ...], bytes]] = []

    def synchronized_commit(observation: _CommitObservation, commit):
        staged_payload = _read_commit_source_bytes(observation)
        source_names = tuple(_commit_source_entry_names(observation))
        if observation.mode == "fd":
            assert source_names == ()
        else:
            assert observation.source_name is not None
            assert source_names == (observation.source_name,)
        assert staged_payload in {first_payload, second_payload}
        assert not _entry_exists(
            output_path.name,
            directory_fd=observation.parent_fd,
        )
        with observation_lock:
            observed_commits.append(
                (observation.mode, source_names, staged_payload)
            )
        commit_barrier.wait(timeout=10)
        return commit()

    def publish(source, payload: bytes) -> tuple[str, str | None, bytes]:
        try:
            api.export_decode_fixture(source, output_path)
        except api.DecodeFixtureError as error:
            return ("error", error.code, payload)
        return ("success", None, payload)

    commit_mode = _install_commit_wrapper(
        api,
        monkeypatch,
        output_parent=output_parent,
        output_name=output_path.name,
        parent_identity=parent_identity,
        handler=synchronized_commit,
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = (
                executor.submit(publish, first_source, first_payload),
                executor.submit(publish, second_source, second_payload),
            )
            results = tuple(future.result(timeout=20) for future in futures)
    finally:
        os.close(parent_descriptor)

    successes = [result for result in results if result[0] == "success"]
    failures = [result for result in results if result[0] == "error"]
    assert len(successes) == 1
    assert len(failures) == 1
    assert failures[0][1] == "output_exists"
    assert output_path.read_bytes() == successes[0][2]
    assert output_path.read_bytes() in {first_payload, second_payload}
    assert len(observed_commits) == 2
    assert {payload for _, _, payload in observed_commits} == {
        first_payload,
        second_payload,
    }
    assert all(mode == commit_mode for mode, _, _ in observed_commits)
    if commit_mode == "fd":
        assert all(not source_names for _, source_names, _ in observed_commits)
    else:
        assert len(
            {source_names[0] for _, source_names, _ in observed_commits}
        ) == 2
    assert tuple(output_parent.iterdir()) == (output_path,)


@pytest.mark.parametrize("existing_kind", ("regular", "symlink"))
def test_export_decode_fixture_refuses_existing_output_without_overwrite(
    tmp_path: Path,
    existing_kind: str,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    source = _make_source(api, seed=20260719)
    output_path = tmp_path / "decode.safetensors"
    if existing_kind == "regular":
        protected_path = output_path
        output_path.write_bytes(b"keep-regular")
    else:
        protected_path = tmp_path / "symlink-target.safetensors"
        protected_path.write_bytes(b"keep-symlink-target")
        output_path.symlink_to(protected_path)
    protected_bytes = protected_path.read_bytes()

    with pytest.raises(api.DecodeFixtureError) as error:
        api.export_decode_fixture(source, output_path)

    assert error.value.code == "output_exists"
    assert protected_path.read_bytes() == protected_bytes
    if existing_kind == "symlink":
        assert output_path.is_symlink()


@pytest.mark.parametrize("unsafe_kind", ("symlinked_parent", "directory_output"))
def test_export_decode_fixture_rejects_unsafe_output_path(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    source = _make_source(api, seed=20260719)
    if unsafe_kind == "symlinked_parent":
        real_parent = tmp_path / "real-parent"
        real_parent.mkdir()
        linked_parent = tmp_path / "linked-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        output_path = linked_parent / "decode.safetensors"
    else:
        real_parent = tmp_path
        output_path = tmp_path / "directory-output"
        output_path.mkdir()
    before = tuple(sorted(path.relative_to(real_parent) for path in real_parent.rglob("*")))

    with pytest.raises(api.DecodeFixtureError) as error:
        api.export_decode_fixture(source, output_path)

    assert error.value.code == "unsafe_output_path"
    after = tuple(sorted(path.relative_to(real_parent) for path in real_parent.rglob("*")))
    assert after == before


def test_export_decode_fixture_serializer_failure_removes_new_parent_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    source = _make_source(api, seed=20260719)
    output_path = tmp_path / "new" / "nested" / "decode.safetensors"
    forced = api.DecodeFixtureError("forced_failure", "forced serializer failure")

    def fail_serializer(_source):
        raise forced

    monkeypatch.setattr(api, "serialize_decode_fixture", fail_serializer)
    with pytest.raises(api.DecodeFixtureError) as error:
        api.export_decode_fixture(source, output_path)

    assert error.value is forced
    assert not os.path.lexists(output_path)
    assert tuple(tmp_path.rglob("*")) == ()


def test_export_decode_fixture_serializer_failure_keeps_existing_empty_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    source = _make_source(api, seed=20260719)
    tensor_snapshots = _snapshot_source_tensors(source.captured)
    output_parent = tmp_path / "publish"
    output_parent.mkdir()
    output_path = output_parent / "decode.safetensors"
    forced = api.DecodeFixtureError("forced_failure", "forced serializer failure")

    def fail_serializer(_source):
        raise forced

    monkeypatch.setattr(api, "serialize_decode_fixture", fail_serializer)
    with pytest.raises(api.DecodeFixtureError) as error:
        api.export_decode_fixture(source, output_path)

    assert error.value is forced
    assert not output_path.exists()
    assert output_parent.is_dir()
    assert tuple(output_parent.iterdir()) == ()
    for tensor, snapshot in tensor_snapshots:
        assert tensor.dtype == snapshot.dtype
        assert tensor.shape == snapshot.shape
        assert torch.equal(tensor, snapshot)


def test_decode_fixture_main_publishes_and_refuses_repeat_overwrite(
    verified_decode_source_bundles,
    tmp_path: Path,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    bundle_root, _, _, lock_path, lock_digest = verified_decode_source_bundles["l3"]
    output_path = tmp_path / "cli" / "decode.safetensors"
    argv = [
        "--bundle",
        str(bundle_root),
        "--publication-lock",
        str(lock_path),
        "--expected-publication-lock-blake3",
        lock_digest,
        "--out",
        str(output_path),
    ]
    direct_source = api.load_decode_fixture_source(
        bundle_root,
        publication_lock_path=lock_path,
        expected_publication_lock_blake3=lock_digest,
    )
    expected = api.serialize_decode_fixture(direct_source)

    assert api.main(argv) == 0
    assert output_path.read_bytes() == expected
    original = output_path.read_bytes()
    assert api.main(argv) == 2
    assert output_path.read_bytes() == original


@pytest.mark.parametrize("argv", ([], ["--unknown-option"]))
def test_decode_fixture_main_uses_argparse_errors(argv: list[str]) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")

    with pytest.raises(SystemExit) as error:
        api.main(argv)

    assert error.value.code == 2


def test_export_decode_fixture_thin_wrapper_runs_end_to_end(
    verified_decode_source_bundles,
    tmp_path: Path,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    bundle_root, _, _, lock_path, lock_digest = verified_decode_source_bundles["l3"]
    output_path = tmp_path / "wrapper" / "decode.safetensors"
    direct_source = api.load_decode_fixture_source(
        bundle_root,
        publication_lock_path=lock_path,
        expected_publication_lock_blake3=lock_digest,
    )
    expected = api.serialize_decode_fixture(direct_source)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "tools/reference_capture"
    command = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        str(REPO_ROOT / "tools" / "reference_capture" / "export_decode_fixture.py"),
        "--bundle",
        str(bundle_root),
        "--publication-lock",
        str(lock_path),
        "--expected-publication-lock-blake3",
        lock_digest,
        "--out",
        str(output_path),
    ]

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    assert output_path.read_bytes() == expected


def test_decode_fixture_module_subprocess_publishes_and_refuses_repeat_overwrite(
    verified_decode_source_bundles,
    tmp_path: Path,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    bundle_root, _, _, lock_path, lock_digest = verified_decode_source_bundles["l3"]
    output_path = tmp_path / "module" / "decode.safetensors"
    direct_source = api.load_decode_fixture_source(
        bundle_root,
        publication_lock_path=lock_path,
        expected_publication_lock_blake3=lock_digest,
    )
    expected = api.serialize_decode_fixture(direct_source)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "tools/reference_capture"
    command = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        "-m",
        "pvlc_reference.decode_fixture",
        "--bundle",
        str(bundle_root),
        "--publication-lock",
        str(lock_path),
        "--expected-publication-lock-blake3",
        lock_digest,
        "--out",
        str(output_path),
    ]

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    assert output_path.read_bytes() == expected
    original = output_path.read_bytes()

    repeated = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        check=False,
    )

    assert repeated.returncode != 0
    assert output_path.read_bytes() == original
    stderr = repeated.stderr.lower()
    assert stderr
    assert b"output" in stderr
    assert b"exist" in stderr


@pytest.mark.parametrize(
    (
        "case_id",
        "mapping_name",
        "semantic_id",
        "mutation",
        "expected_code",
    ),
    TENSOR_CONTRACT_CASES,
    ids=[case[0] for case in TENSOR_CONTRACT_CASES],
)
def test_serialize_decode_fixture_rejects_each_malformed_tensor_contract(
    case_id: str,
    mapping_name: str,
    semantic_id: str,
    mutation: str,
    expected_code: str,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    corrupted = _malform_source_tensor(
        _tensor_contract_source(),
        mapping_name,
        semantic_id,
        mutation,
    )

    with pytest.raises(api.DecodeFixtureError) as error:
        api.serialize_decode_fixture(corrupted)

    assert error.value.code == expected_code, case_id


@pytest.mark.parametrize(
    ("case", "expected_code"),
    (
        ("missing_tensor", "missing_tensor"),
        ("cache_rank", "invalid_tensor_shape"),
        ("cache_sequence", "invalid_tensor_shape"),
        ("cache_dtype", "invalid_tensor_dtype"),
        ("cache_nonfinite", "nonfinite_tensor"),
        ("cache_prefix_drift", "cache_prefix_mismatch"),
        ("layer0_append_mismatch", "cache_append_mismatch"),
        ("layer0_value_append_mismatch", "cache_append_mismatch"),
        ("mask_nonbinary", "invalid_attention_mask"),
        ("mask_wrong_prefix", "invalid_attention_mask"),
        ("mask_appended_zero", "invalid_attention_mask"),
        ("cache_position", "cache_position_mismatch"),
        ("position_axis", "position_id_mismatch"),
        ("rope_delta", "rope_delta_mismatch"),
        ("trace_token", "invalid_generation_trace"),
        ("trace_input_chain", "invalid_generation_trace"),
        ("publication_case", "invalid_source_publication"),
        ("publication_text", "invalid_source_publication"),
        ("publication_repeat3", "invalid_source_publication"),
        ("publication_repeat_bool", "insufficient_repeat_count"),
        ("publication_repeat_nonint", "insufficient_repeat_count"),
        ("missing_trace", "invalid_generation_trace"),
        ("alternate_tokens", "invalid_source_publication"),
        ("processor_terminal", "invalid_generation_trace"),
        ("prefill_logits_argmax", "invalid_generation_trace"),
        ("trace_kv_shapes", "cache_shape_mismatch"),
        ("mrope_terminal_position", "position_id_mismatch"),
        ("logits_argmax", "logits_argmax_mismatch"),
        ("pipeline_shape", "invalid_tensor_shape"),
        ("pipeline_dtype", "invalid_tensor_dtype"),
        ("pipeline_nonfinite", "nonfinite_tensor"),
        ("logits_shape", "invalid_tensor_shape"),
        ("logits_dtype", "invalid_tensor_dtype"),
        ("logits_nonfinite", "nonfinite_tensor"),
        ("provenance_device", "invalid_provenance"),
        ("provenance_dtype", "invalid_provenance"),
        ("provenance_revision", "invalid_provenance"),
        ("provenance_version", "invalid_provenance"),
        ("provenance_model_lock", "invalid_provenance"),
        ("provenance_capture_tool", "invalid_provenance"),
        ("provenance_python", "invalid_provenance"),
        ("provenance_torch", "invalid_provenance"),
        ("provenance_transformers", "invalid_provenance"),
        ("provenance_shim", "invalid_provenance"),
        ("provenance_nondeterministic", "invalid_provenance"),
        ("repeat_count", "insufficient_repeat_count"),
        ("source_digest", "invalid_source_digest"),
        ("source_fingerprint", "invalid_source_digest"),
        ("publication_lock_digest_malformed", "invalid_source_digest"),
        (
            "publication_lock_digest_wrong",
            "publication_lock_digest_mismatch",
        ),
        (
            "publication_stale_lock_after_fingerprint",
            "publication_lock_digest_mismatch",
        ),
    ),
    ids=lambda value: value,
)
def test_serialize_decode_fixture_rejects_corrupted_verified_source(
    case: str,
    expected_code: str,
) -> None:
    api = importlib.import_module("pvlc_reference.decode_fixture")
    corrupted = _corrupt_source(_make_source(api, seed=20260719), case)

    with pytest.raises(api.DecodeFixtureError) as error:
        api.serialize_decode_fixture(corrupted)

    assert error.value.code == expected_code
