from __future__ import annotations

"""Batch runner for the self-contained Full Pipeline submission workflow."""

import argparse
import os
import traceback
from pathlib import Path
import sys

FULL_PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FULL_PIPELINE_DIR.parent
# Several legacy modules still assume they are executed from the Full Pipeline
# folder and resolve assets with relative paths, so the batch runner anchors
# the process there before importing the rest of the pipeline.
os.chdir(FULL_PIPELINE_DIR)

full_pipeline_path = str(FULL_PIPELINE_DIR)
if full_pipeline_path not in sys.path:
    sys.path.insert(0, full_pipeline_path)

from axes_extraction import run_axes
from edge_detection import build_adjacency, build_graph_from_skeleton
from extract_residual_edges import recover_leftover_graph
from face_extraction import extract_mesh_elements
from global_turns_detection import check_edges_global_curvature, split_edges_on_turns
from nas_export import write_nas_from_pipeline
from nas_visualize import save_nas_mesh_preview, save_nas_overlay_preview
from node_extraction import get_skeleton_nodes
from nodes_classification import classify_all_nodes
from utils import filter_nodes_keep_indices, open_image


def build_parser() -> argparse.ArgumentParser:
    """Expose the main knobs needed for competition batch runs."""
    full_pipeline_dir = Path(__file__).resolve().parent
    project_root = full_pipeline_dir.parent
    default_input_dir = project_root / "jpeg images"
    default_output_dir = project_root / "submission_outputs_full_pipeline"

    parser = argparse.ArgumentParser(
        description="Run the Full Pipeline over all competition images without stopping on failures.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir,
        help="Directory containing competition JPEG images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory where NAS files and previews will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional max number of images to process.",
    )
    return parser


def _iter_images(input_dir: Path, limit: int | None = None) -> list[Path]:
    """Return the sorted competition image list, optionally truncated for testing."""
    images = sorted(path for path in input_dir.glob("*.jpg") if path.is_file())
    if limit is not None:
        images = images[:limit]
    return images


def process_image(image_path: Path, output_dir: Path) -> tuple[bool, str]:
    """Run one full image through graph recovery, face extraction, and export."""
    image = open_image(image_path)
    axes_result = run_axes(image_path)

    skeleton, nodes, _, _ = get_skeleton_nodes(
        image,
        trace_endpoints=True,
        trace_snap_radius=3,
        trace_max_extension=25,
        trace_search_radius=2,
    )

    confirmed_nodes, rejected_nodes, _ = classify_all_nodes(skeleton, nodes)
    print(f"[{image_path.name}] confirmed_nodes={len(confirmed_nodes)} total_nodes={len(nodes)}")
    print(f"[{image_path.name}] rejected_nodes={rejected_nodes}")

    graph = build_graph_from_skeleton(skeleton, confirmed_nodes)
    print(f"[{image_path.name}] initial_edges={len(graph['edges'])}")

    recovered = recover_leftover_graph(
        skeleton=skeleton,
        old_nodes=graph["nodes"],
        old_edges=graph["edges"],
        old_edge_paths=graph["edge_paths"],
        match_radius=20,
        classify_candidates=False,
        old_node_mask_radius=6,
        remove_end_neighborhood=0,
    )
    print(f"[{image_path.name}] recovered_new_edges={recovered['mapping_info']['num_new_edges_added']}")

    subset_recovered = {
        "nodes": recovered["merged_nodes"],
        "edges": recovered["merged_edges"],
        "edge_paths": recovered["merged_edge_paths"],
    }

    curvature = check_edges_global_curvature(
        subset_recovered["edge_paths"],
        direction_window=12,
        comparison_step=6,
        angle_threshold_deg=13.0,
        min_peak_gap=10,
        ignore_ends=4,
    )
    graph_new = split_edges_on_turns(
        subset_recovered,
        curvature,
        merge_dist=15,
        dbscan_eps=8,
        dbscan_min_samples=2,
        snap_path_endpoints=True,
    )

    graph_new["nodes"], rejected_nodes, _ = classify_all_nodes(
        skeleton,
        graph_new["nodes"],
        outer_radius=20,
        inner_radius=4,
        direction_step=5,
        angle_tol_deg=10,
        min_branch_length=3,
        min_branches_for_node=2,
        round_input_nodes=True,
        return_debug=False,
    )
    graph_new["node_map"] = filter_nodes_keep_indices(graph_new["nodes"], graph_new["node_map"])
    graph_new["adjacency"] = build_adjacency(graph_new["edges"], len(graph_new["node_map"]))

    # Face extraction is the stage most likely to fail on a bad intermediate
    # graph, so it gets its own narrow failure path and does not stop the batch.
    try:
        triangles, quads = extract_mesh_elements(graph_new["node_map"], graph_new["edges"])
    except Exception as exc:
        message = f"face detection failed: {exc}"
        print(f"[{image_path.name}] {message}")
        return False, message

    if not triangles and not quads:
        message = "face detection returned no triangles or quads"
        print(f"[{image_path.name}] {message}")
        return False, message

    nas_output_path = output_dir / f"{image_path.stem}.nas"
    write_nas_from_pipeline(
        graph_new["node_map"],
        triangles,
        quads,
        image_path=image_path,
        origin_point=axes_result["intersection"],
        x_scale=axes_result["x_scale"],
        y_scale=axes_result["y_scale"],
        output_path=nas_output_path,
    )

    preview_dir = output_dir / "previews"
    mesh_preview_path = preview_dir / f"{image_path.stem}_mesh.png"
    overlay_preview_path = preview_dir / f"{image_path.stem}_overlay.png"

    save_nas_mesh_preview(nas_output_path, mesh_preview_path, show_node_ids=True)
    save_nas_overlay_preview(
        nas_output_path,
        image_path=image_path,
        output_path=overlay_preview_path,
        origin_point=axes_result["intersection"],
        x_scale=axes_result["x_scale"],
        y_scale=axes_result["y_scale"],
        show_node_ids=True,
    )

    message = (
        f"wrote {nas_output_path.name} "
        f"with {len(quads)} quads and {len(triangles)} triangles"
    )
    print(f"[{image_path.name}] {message}")
    return True, message


def main() -> None:
    """Process the whole competition folder while preserving per-image failures."""
    args = build_parser().parse_args()
    images = _iter_images(args.input_dir, args.limit)
    if not images:
        raise SystemExit(f"No JPG images found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    successes = 0

    for index, image_path in enumerate(images, start=1):
        print(f"\n[{index}/{len(images)}] Processing {image_path.name}")
        try:
            ok, message = process_image(image_path, args.output_dir)
        except Exception as exc:
            ok = False
            message = f"{type(exc).__name__}: {exc}"
            print(f"[{image_path.name}] unexpected failure")
            print(traceback.format_exc())

        if ok:
            successes += 1
        else:
            failures.append((image_path.name, message))

    print("\nBatch run finished.")
    print(f"Successful images: {successes}")
    print(f"Failed images: {len(failures)}")

    if failures:
        failure_log_path = args.output_dir / "failed_images.txt"
        failure_log_path.write_text(
            "\n".join(f"{name}: {message}" for name, message in failures) + "\n",
            encoding="utf-8",
        )
        print(f"Failure log written to {failure_log_path}")


if __name__ == "__main__":
    main()
