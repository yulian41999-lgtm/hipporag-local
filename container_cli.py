"""Container commands for preparation, indexing, QA, export and evaluation."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent


def resolve_command(arguments):
    if not arguments:
        arguments = ["embedding"]
    command, *rest = arguments
    if command in {"prepare", "index", "answer", "smoke"}:
        return ROOT / "business.py", [command, *rest]
    if command == "evaluate":
        return ROOT / "evaluation/run.py", rest
    if command == "export":
        return ROOT / "data/banking/银行监管测试集/tools/export_eval_input.py", rest
    if command == "probe":
        return ROOT / "probe_llm.py", rest
    if command in {"embedding", "sample", "reload"}:
        return ROOT / "run.py", [command, *rest]
    raise ValueError("Commands: embedding, probe, prepare, index, answer, export, evaluate, sample, reload, smoke")


if __name__ == "__main__":
    if sys.argv[1:] in [["--help"], ["-h"]]:
        print("Commands: embedding | probe | prepare | index | answer | export [arguments] | evaluate [arguments] | sample | reload | smoke")
    else:
        target, arguments = resolve_command(sys.argv[1:])
        if not target.is_file():
            raise FileNotFoundError("Missing command resource: " + str(target))
        sys.argv = [str(target), *arguments]
        runpy.run_path(str(target), run_name="__main__")
