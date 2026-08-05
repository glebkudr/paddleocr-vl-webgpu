import argparse
import ast
import importlib.util
from pathlib import Path

import pytest
from blake3 import blake3


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "tools" / "m7p_mps_native_benchmark.py"


def source_and_tree() -> tuple[str, ast.Module]:
    source = BENCHMARK.read_text(encoding="utf-8")
    return source, ast.parse(source)


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("m7p_fp16_benchmark_contract", BENCHMARK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def main_function(tree: ast.Module) -> ast.FunctionDef:
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    ]
    assert len(functions) == 1
    return functions[0]


def test_native_benchmark_requires_the_shared_converted_checkpoint() -> None:
    source, tree = source_and_tree()
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
    ]
    checkpoint_calls = [
        call
        for call in calls
        if call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "--checkpoint"
    ]
    assert len(checkpoint_calls) == 1
    keywords = {
        keyword.arg: keyword.value
        for keyword in checkpoint_calls[0].keywords
        if keyword.arg is not None
    }
    assert isinstance(keywords.get("required"), ast.Constant)
    assert keywords["required"].value is True
    assert "SNAPSHOT =" not in source


def test_native_benchmark_loads_builtin_transformers_fp16_without_remote_code() -> None:
    module = load_benchmark_module()
    source, tree = source_and_tree()
    main = main_function(tree)

    model_assignments = [
        statement
        for statement in main.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "model"
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "load_fp16_model"
    ]
    assert len(model_assignments) == 1
    load_call = model_assignments[0].value
    assert [dotted_name(argument) for argument in load_call.args] == ["args.checkpoint"]
    assert not load_call.keywords

    from_pretrained_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_pretrained"
        and dotted_name(node.func.value) == "model_class"
    ]
    assert len(from_pretrained_calls) == 1
    keywords = {
        keyword.arg: keyword.value
        for keyword in from_pretrained_calls[0].keywords
        if keyword.arg is not None
    }
    assert dotted_name(keywords["dtype"]) == "torch.float16"
    assert isinstance(keywords["trust_remote_code"], ast.Constant)
    assert keywords["trust_remote_code"].value is False
    assert isinstance(keywords["attn_implementation"], ast.Constant)
    assert keywords["attn_implementation"].value == "sdpa"
    assert "install_transformers_compat" not in source
    assert "trust_remote_code=True" not in source
    assert all(
        dotted_name(node) != "torch.bfloat16"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    )
    assert "torch.bfloat16" not in source

    checkpoint = Path("/contract/shared-fp16/model.safetensors")
    dtype_sentinel = object()
    model_sentinel = object()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class FakeModelClass:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append((args, kwargs))
            return model_sentinel

    class FakeTorch:
        float16 = dtype_sentinel

    result = module.load_fp16_model(
        checkpoint,
        torch_module=FakeTorch,
        model_class=FakeModelClass,
    )
    assert result is model_sentinel
    assert calls == [
        (
            (str(checkpoint.parent),),
            {
                "dtype": dtype_sentinel,
                "trust_remote_code": False,
                "attn_implementation": "sdpa",
            },
        )
    ]


def test_native_benchmark_accepts_a_sota_region_and_task_prompt() -> None:
    _source, tree = source_and_tree()
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ]
    options = {call.args[0].value: call for call in calls}
    assert {"--image", "--bbox", "--prompt"} <= options.keys()

    bbox_keywords = {
        keyword.arg: keyword.value
        for keyword in options["--bbox"].keywords
        if keyword.arg is not None
    }
    assert dotted_name(bbox_keywords["type"]) == "parse_bbox"


def test_sota_chat_template_and_bbox_crop_are_exact() -> None:
    module = load_benchmark_module()
    assert module.render_chat("Table Recognition:") == (
        "<|begin_of_sentence|>User: "
        "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>"
        "Table Recognition:\nAssistant:\n"
    )
    assert module.parse_bbox("113,43,871,508") == (113, 43, 871, 508)

    class FakeImage:
        def __init__(self):
            self.calls = []

        def crop(self, bbox):
            self.calls.append(bbox)
            return "cropped"

    image = FakeImage()
    assert module.crop_image(image, (113, 43, 871, 508)) == "cropped"
    assert image.calls == [(113, 43, 871, 508)]


@pytest.mark.parametrize(
    "value",
    ["", "1,2,3", "1,2,3,4,5", "4,2,1,8", "1,8,4,2", "a,2,3,4"],
)
def test_bbox_parser_rejects_invalid_regions(value: str) -> None:
    module = load_benchmark_module()
    with pytest.raises((ValueError, argparse.ArgumentTypeError)):
        module.parse_bbox(value)


def test_report_authenticates_checkpoint_and_precision_before_timings() -> None:
    _source, tree = source_and_tree()
    main = main_function(tree)

    identity_assignments = [
        statement
        for statement in main.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "identity"
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "checkpoint_identity"
        and [dotted_name(argument) for argument in statement.value.args]
        == ["args.checkpoint"]
        and not statement.value.keywords
    ]
    assert len(identity_assignments) == 1
    identity_assignment = identity_assignments[0]

    identity_only_branches = [
        statement
        for statement in main.body
        if isinstance(statement, ast.If)
        and dotted_name(statement.test) == "args.identity_only"
    ]
    assert len(identity_only_branches) == 1
    identity_only = identity_only_branches[0]
    assert len(identity_only.body) == 2
    print_statement, return_statement = identity_only.body
    assert isinstance(print_statement, ast.Expr)
    assert isinstance(print_statement.value, ast.Call)
    assert dotted_name(print_statement.value.func) == "print"
    assert len(print_statement.value.args) == 1
    json_call = print_statement.value.args[0]
    assert isinstance(json_call, ast.Call)
    assert dotted_name(json_call.func) == "json.dumps"
    assert [dotted_name(argument) for argument in json_call.args] == ["identity"]
    assert isinstance(return_statement, ast.Return)
    assert return_statement.value is None
    identity_options = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--identity-only"
    ]
    assert len(identity_options) == 1
    identity_option_keywords = {
        keyword.arg: keyword.value
        for keyword in identity_options[0].keywords
        if keyword.arg is not None
    }
    assert isinstance(identity_option_keywords.get("action"), ast.Constant)
    assert identity_option_keywords["action"].value == "store_true"

    report_assignments = [
        statement
        for statement in main.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "report"
        and isinstance(statement.value, ast.Dict)
    ]
    assert len(report_assignments) == 1
    report_assignment = report_assignments[0]
    identity_spreads = [
        value
        for key, value in zip(
            report_assignment.value.keys,
            report_assignment.value.values,
            strict=True,
        )
        if key is None and dotted_name(value) == "identity"
    ]
    assert len(identity_spreads) == 1

    model_assignments = [
        statement
        for statement in main.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "model"
    ]
    assert len(model_assignments) == 1
    timing_calls = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and dotted_name(node.func) == "time.perf_counter"
    ]
    mps_availability_calls = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and dotted_name(node.func) == "torch.backends.mps.is_available"
    ]
    assert timing_calls
    assert len(mps_availability_calls) == 1
    assert identity_assignment.lineno < identity_only.lineno
    assert identity_only.lineno < mps_availability_calls[0].lineno
    assert identity_only.lineno < model_assignments[0].lineno
    assert identity_assignment.lineno < min(call.lineno for call in timing_calls)

    report_writes = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "write_text"
        and dotted_name(node.func.value) == "args.out"
        and node.args
        and isinstance(node.args[0], ast.Call)
        and dotted_name(node.args[0].func) == "json.dumps"
        and node.args[0].args
        and dotted_name(node.args[0].args[0]) == "report"
    ]
    assert len(report_writes) == 1


def test_checkpoint_identity_hashes_the_exact_resolved_model_file(tmp_path: Path) -> None:
    module = load_benchmark_module()
    checkpoint = tmp_path / "model.safetensors"
    payload = bytes((index * 97 + index // 13 + 7) % 256 for index in range(8193))
    checkpoint.write_bytes(payload)

    assert module.checkpoint_identity(checkpoint) == {
        "checkpoint_path": str(checkpoint.resolve(strict=True)),
        "checkpoint_blake3": blake3(payload).hexdigest(),
        "checkpoint_bytes": len(payload),
        "dtype": "float16",
    }


def test_checkpoint_identity_rejects_a_sibling_transformers_would_not_load(
    tmp_path: Path,
) -> None:
    module = load_benchmark_module()
    (tmp_path / "model.safetensors").write_bytes(b"accepted")
    sibling = tmp_path / "other.safetensors"
    sibling.write_bytes(b"wrong")

    with pytest.raises(ValueError, match=r"exactly model\.safetensors"):
        module.checkpoint_identity(sibling)
