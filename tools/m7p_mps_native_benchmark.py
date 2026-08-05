#!/usr/bin/env python3
"""Native PyTorch/MPS benchmark for PaddleOCR-VL-1.6.

Drives one explicitly supplied converted FP16 checkpoint through the built-in
Transformers PaddleOCR-VL implementation and records, per case and per run:

- processor (host preprocessing) wall time;
- vision encoder + projector time inside the first generation forward;
- decoder prefill time (first forward minus the vision window);
- per-token decode forward times (each subsequent generate forward);
- end-to-end generate time and the generated text.

An arbitrary image, crop, and task prompt can be supplied so the native path
can reproduce the exact per-layout-region contract used by PaddleOCR-VL's
official page pipeline.

Usage:
  .venv/bin/python tools/m7p_mps_native_benchmark.py \
      --checkpoint models/fp16/.../model.safetensors \
      --image invoice.jpg --bbox 113,43,871,508 \
      --prompt "Table Recognition:" --runs 1 --max-new 1024 \
      [--out output/benchmark/m7p-mps-native.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from blake3 import blake3

ROOT = Path(__file__).resolve().parent.parent
CASES = [
    {
        "case": "ocr.clean_latin.0001",
        "image": ROOT / "cases" / "smoke" / "assets" / "ocr-clean-latin.png",
        "prompt": "OCR:",
    },
    {
        "case": "smoke.chart_bars",
        "image": ROOT / "cases" / "smoke" / "assets" / "chart-bars.png",
        "prompt": "OCR:",
    },
    {
        "case": "smoke.formula_indices",
        "image": ROOT / "cases" / "smoke" / "assets" / "formula-indices.png",
        "prompt": "OCR:",
    },
]


def render_chat(prompt: str) -> str:
    """Render the exact PaddleOCR-VL chat template used by the Sota pipeline."""

    return (
        "<|begin_of_sentence|>User: "
        "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>"
        f"{prompt}\nAssistant:\n"
    )


def parse_bbox(value: str) -> tuple[int, int, int, int]:
    try:
        coordinates = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "bbox must contain four comma-separated integers"
        ) from error
    if len(coordinates) != 4:
        raise argparse.ArgumentTypeError(
            "bbox must contain exactly four comma-separated integers"
        )
    left, top, right, bottom = coordinates
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise argparse.ArgumentTypeError(
            "bbox must satisfy 0 <= left < right and 0 <= top < bottom"
        )
    return left, top, right, bottom


def crop_image(image, bbox: tuple[int, int, int, int] | None):
    return image if bbox is None else image.crop(bbox)


def checkpoint_identity(checkpoint: Path) -> dict[str, str | int]:
    if checkpoint.name != "model.safetensors":
        raise ValueError("checkpoint path must name exactly model.safetensors")
    resolved = checkpoint.resolve(strict=True)
    digest = blake3()
    byte_count = 0
    with resolved.open("rb") as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
            byte_count += len(block)
    return {
        "checkpoint_path": str(resolved),
        "checkpoint_blake3": digest.hexdigest(),
        "checkpoint_bytes": byte_count,
        "dtype": "float16",
    }


def load_fp16_model(
    checkpoint: Path,
    torch_module=torch,
    model_class=None,
):
    torch = torch_module
    if model_class is None:
        from transformers import AutoModelForImageTextToText

        model_class = AutoModelForImageTextToText
    return model_class.from_pretrained(
        str(checkpoint.parent),
        dtype=torch.float16,
        trust_remote_code=False,
        attn_implementation="sdpa",
    )


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * fraction))
    return ordered[index]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "median_ms": statistics.median(values) * 1e3,
        "min_ms": min(values) * 1e3,
        "max_ms": max(values) * 1e3,
        "mean_ms": statistics.mean(values) * 1e3,
    }


def summarize_rates(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--identity-only", action="store_true")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--bbox", type=parse_bbox)
    parser.add_argument("--prompt", default="OCR:")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "benchmark" / "m7p-mps-native.json")
    args = parser.parse_args()

    identity = checkpoint_identity(args.checkpoint)
    if args.identity_only:
        print(json.dumps(identity))
        return

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS device is unavailable")
    if args.bbox is not None and args.image is None:
        parser.error("--bbox requires --image")
    if args.runs < 1 or args.warmup < 0 or args.max_new < 1:
        parser.error("--runs and --max-new must be positive; --warmup cannot be negative")

    from transformers import AutoProcessor

    checkpoint_directory = args.checkpoint.resolve(strict=True).parent
    processor = AutoProcessor.from_pretrained(
        checkpoint_directory, trust_remote_code=False
    )
    model = load_fp16_model(args.checkpoint)
    model.to("mps")
    model.eval()

    if args.image is None:
        runtime_cases = [{**case, "bbox": None} for case in CASES]
    else:
        image_path = args.image.resolve(strict=True)
        runtime_cases = [
            {
                "case": image_path.stem,
                "image": image_path,
                "prompt": args.prompt,
                "bbox": args.bbox,
            }
        ]

    forward_starts: list[float] = []
    forward_windows: list[float] = []
    vision_starts: list[float] = []
    vision_windows: list[float] = []

    def forward_pre(_module, _arguments):
        torch.mps.synchronize()
        forward_starts.append(time.perf_counter())

    def forward_post(_module, _arguments, _output):
        torch.mps.synchronize()
        forward_windows.append(time.perf_counter() - forward_starts.pop())

    def vision_pre(_module, _arguments):
        torch.mps.synchronize()
        vision_starts.append(time.perf_counter())

    def projector_post(_module, _arguments, _output):
        torch.mps.synchronize()
        vision_windows.append(time.perf_counter() - vision_starts.pop())

    hooks = [
        model.register_forward_pre_hook(forward_pre),
        model.register_forward_hook(forward_post),
        model.model.visual.register_forward_pre_hook(vision_pre),
        model.model.projector.register_forward_hook(projector_post),
    ]

    report = {
        **identity,
        "device": "mps",
        "torch": torch.__version__,
        "transformers_model_class": type(model).__name__,
        "model_source": "transformers_builtin",
        "trust_remote_code": False,
        "attention_implementation": "sdpa",
        "max_new_tokens": args.max_new,
        "cases": [],
    }

    try:
        for case in runtime_cases:
            from PIL import Image

            t_image_start = time.perf_counter()
            image = Image.open(case["image"]).convert("RGB")
            source_size = image.size
            if case["bbox"] is not None:
                _left, _top, right, bottom = case["bbox"]
                if right > image.width or bottom > image.height:
                    raise ValueError(
                        f"bbox {case['bbox']} exceeds image size {image.size}"
                    )
            image = crop_image(image, case["bbox"])
            image_load_seconds = time.perf_counter() - t_image_start
            chat = render_chat(case["prompt"])
            runs = []
            for iteration in range(args.warmup + args.runs):
                with torch.no_grad():
                    t_processor_start = time.perf_counter()
                    processed = processor(
                        text=chat,
                        images=image,
                        return_tensors="pt",
                    )
                    inputs = {
                        key: value.to("mps") if hasattr(value, "to") else value
                        for key, value in processed.items()
                    }
                    torch.mps.synchronize()
                    processor_seconds = time.perf_counter() - t_processor_start

                    forward_starts.clear()
                    forward_windows.clear()
                    vision_starts.clear()
                    vision_windows.clear()
                    model.model.rope_deltas = None
                    t_generate_start = time.perf_counter()
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new,
                        do_sample=False,
                        use_cache=True,
                    )
                    torch.mps.synchronize()
                    generate_seconds = time.perf_counter() - t_generate_start
                    prompt_length = int(inputs["input_ids"].shape[-1])
                    token_ids = generated[0, prompt_length:].tolist()
                    token_count = len(token_ids)
                    text = processor.tokenizer.decode(
                        token_ids,
                        skip_special_tokens=True,
                    )
                    vision_seconds = vision_windows[0] if vision_windows else None
                    prefill_total_seconds = (
                        forward_windows[0] if forward_windows else None
                    )
                    decoder_prefill_seconds = (
                        max(0.0, prefill_total_seconds - vision_seconds)
                        if prefill_total_seconds is not None
                        and vision_seconds is not None
                        else None
                    )
                    decode_windows = forward_windows[1:]
                    decode_tokens_per_second = (
                        len(decode_windows) / sum(decode_windows)
                        if decode_windows
                        else None
                    )
                    runs.append(
                        {
                            "warmup": iteration < args.warmup,
                            "processor_s": processor_seconds,
                            "vision_s": vision_seconds,
                            "prefill_total_s": prefill_total_seconds,
                            "decoder_prefill_s": decoder_prefill_seconds,
                            "decode_steps_s": decode_windows,
                            "decode_tokens_per_second": decode_tokens_per_second,
                            "generation_tokens_per_second": (
                                token_count / generate_seconds
                            ),
                            "generate_s": generate_seconds,
                            "end_to_end_s": processor_seconds + generate_seconds,
                            "new_tokens": token_count,
                            "token_ids": token_ids,
                            "text": text,
                        }
                    )
            timed = [run for run in runs if not run["warmup"]]
            decode_all = [
                value for run in timed for value in run["decode_steps_s"]
            ]
            case_report = {
                "case": case["case"],
                "image": str(Path(case["image"]).resolve(strict=True)),
                "source_size": list(source_size),
                "input_size": list(image.size),
                "bbox": list(case["bbox"]) if case["bbox"] is not None else None,
                "prompt": case["prompt"],
                "chat": chat,
                "image_load_s": image_load_seconds,
                "image_grid_thw": inputs["image_grid_thw"].detach().cpu().tolist(),
                "prompt_tokens": int(inputs["input_ids"].shape[1]),
                "new_tokens": timed[-1]["new_tokens"],
                "token_ids": timed[-1]["token_ids"],
                "text": timed[-1]["text"],
                "processor": summarize([run["processor_s"] for run in timed]),
                "vision": summarize(
                    [run["vision_s"] for run in timed if run["vision_s"] is not None]
                ),
                "prefill_total": summarize(
                    [
                        run["prefill_total_s"]
                        for run in timed
                        if run["prefill_total_s"] is not None
                    ]
                ),
                "decoder_prefill": summarize(
                    [
                        run["decoder_prefill_s"]
                        for run in timed
                        if run["decoder_prefill_s"] is not None
                    ]
                ),
                "decode_per_token": (
                    summarize(decode_all) if decode_all else None
                ),
                "decode_tokens_per_second": summarize_rates(
                    [
                        run["decode_tokens_per_second"]
                        for run in timed
                        if run["decode_tokens_per_second"] is not None
                    ]
                ),
                "generation_tokens_per_second": summarize_rates(
                    [run["generation_tokens_per_second"] for run in timed]
                ),
                "generate_total": summarize([run["generate_s"] for run in timed]),
                "end_to_end": summarize([run["end_to_end_s"] for run in timed]),
            }
            report["cases"].append(case_report)
            print(json.dumps(case_report, ensure_ascii=False))
    finally:
        for hook in hooks:
            hook.remove()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
