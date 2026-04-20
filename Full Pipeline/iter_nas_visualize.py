from __future__ import annotations

"""Batch-regenerate clean and overlay previews from existing NAS outputs."""

import argparse
from pathlib import Path

from axes_extraction import run_axes
from nas_visualize import save_nas_mesh_preview, save_nas_overlay_preview


def build_parser() -> argparse.ArgumentParser:
    """CLI arguments for preview regeneration runs."""
    full_pipeline_dir = Path(__file__).resolve().parent
    project_root = full_pipeline_dir.parent

    parser = argparse.ArgumentParser(
        description="Batch-render clean NAS previews and source-image overlays.",
    )
    parser.add_argument(
        "--nas-dir",
        type=Path,
        default=project_root / "submission_outputs_full_pipeline",
        help="Directory containing generated .nas files.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=project_root / "jpeg images",
        help="Directory containing the original JPEG inputs.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=project_root / "submission_outputs_full_pipeline" / "previews",
        help="Directory where preview PNGs will be saved.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional max number of NAS files to process.",
    )
    return parser


def _iter_nas_files(nas_dir: Path, limit: int | None = None) -> list[Path]:
    """Return the generated NAS files to visualize."""
    files = sorted(path for path in nas_dir.glob("*.nas") if path.is_file())
    if limit is not None:
        files = files[:limit]
    return files


def process_nas_file(nas_path: Path, image_dir: Path, preview_dir: Path) -> tuple[bool, str]:
    """Render one NAS file as both a clean mesh preview and a source-image overlay."""
    stem = nas_path.stem
    image_path = image_dir / f"{stem}.jpg"
    mesh_preview_path = preview_dir / f"{stem}_mesh.png"
    overlay_preview_path = preview_dir / f"{stem}_overlay.png"

    save_nas_mesh_preview(nas_path, mesh_preview_path, show_node_ids=True)

    if not image_path.is_file():
        return False, f"missing source image for overlay: {image_path.name}"

    axes_result = run_axes(image_path)
    save_nas_overlay_preview(
        nas_path,
        image_path=image_path,
        output_path=overlay_preview_path,
        origin_point=axes_result["intersection"],
        x_scale=axes_result["x_scale"],
        y_scale=axes_result["y_scale"],
        show_node_ids=True,
    )
    return True, f"saved {mesh_preview_path.name} and {overlay_preview_path.name}"


def main() -> None:
    """Process every NAS file in the output folder without aborting on failures."""
    args = build_parser().parse_args()
    nas_files = _iter_nas_files(args.nas_dir, args.limit)
    if not nas_files:
        raise SystemExit(f"No .nas files found in {args.nas_dir}")

    args.preview_dir.mkdir(parents=True, exist_ok=True)

    successes = 0
    failures: list[tuple[str, str]] = []

    for index, nas_path in enumerate(nas_files, start=1):
        print(f"\n[{index}/{len(nas_files)}] Visualizing {nas_path.name}")
        try:
            ok, message = process_nas_file(nas_path, args.image_dir, args.preview_dir)
        except Exception as exc:
            ok = False
            message = f"{type(exc).__name__}: {exc}"

        if ok:
            successes += 1
            print(f"[{nas_path.name}] {message}")
        else:
            failures.append((nas_path.name, message))
            print(f"[{nas_path.name}] failed: {message}")

    print("\nVisualization batch finished.")
    print(f"Successful files: {successes}")
    print(f"Failed files: {len(failures)}")

    if failures:
        failure_log_path = args.preview_dir / "failed_visualizations.txt"
        failure_log_path.write_text(
            "\n".join(f"{name}: {message}" for name, message in failures) + "\n",
            encoding="utf-8",
        )
        print(f"Failure log written to {failure_log_path}")


if __name__ == "__main__":
    main()
