from __future__ import annotations

"""Submission entrypoint that delegates to the self-contained Full Pipeline."""

import importlib.util
from pathlib import Path


def _load_full_pipeline_main():
    """Load the batch pipeline from the folder with a space in its name."""
    script_path = Path(__file__).resolve().parent / "Full Pipeline" / "iter_pipeline.py"
    spec = importlib.util.spec_from_file_location("full_pipeline_iter", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Full Pipeline entrypoint from {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def main() -> None:
    """Run the Full Pipeline batch entrypoint exactly as the brief expects."""
    full_pipeline_main = _load_full_pipeline_main()
    full_pipeline_main()


if __name__ == "__main__":
    main()
