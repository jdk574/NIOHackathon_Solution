from __future__ import annotations

"""Utilities for rendering exported NAS meshes as standalone or overlay previews."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_nas_file(nas_path: str | Path):
    """Read GRID / CTRIA3 / CQUAD4 entries into simple Python structures."""
    nas_path = Path(nas_path)
    nodes: dict[int, tuple[float, float]] = {}
    triangles: list[tuple[int, int, int]] = []
    quads: list[tuple[int, int, int, int]] = []

    for raw_line in nas_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line in {"BEGIN BULK", "ENDDATA"}:
            continue

        parts = line.split()
        card = parts[0]

        if card == "GRID":
            if len(parts) >= 5:
                node_id = int(parts[1])
                x_value = float(parts[2])
                y_value = float(parts[3])
            else:
                fields = [raw_line[i:i + 8] for i in range(0, len(raw_line), 8)]
                if len(fields) < 6:
                    raise ValueError(f"Invalid GRID line: {raw_line}")
                node_id = int(fields[1].strip())
                x_value = float(fields[3].strip())
                y_value = float(fields[4].strip())
            nodes[node_id] = (x_value, y_value)
        elif card == "CTRIA3":
            if len(parts) < 6:
                raise ValueError(f"Invalid CTRIA3 line: {raw_line}")
            triangles.append((int(parts[3]), int(parts[4]), int(parts[5])))
        elif card == "CQUAD4":
            if len(parts) < 7:
                raise ValueError(f"Invalid CQUAD4 line: {raw_line}")
            quads.append((int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])))

    if not nodes:
        raise ValueError(f"No GRID nodes found in {nas_path}")

    return nodes, triangles, quads


def _to_canvas_transform(
    nodes: dict[int, tuple[float, float]],
    image_size: int,
    padding: int,
):
    """Create a simple model-space-to-canvas transform for clean preview images."""
    coords = np.array(list(nodes.values()), dtype=float)
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)

    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    scale = min(
        (image_size - 2 * padding) / span_x,
        (image_size - 2 * padding) / span_y,
    )

    def transform(point_xy: tuple[float, float]) -> tuple[int, int]:
        x_value, y_value = point_xy
        canvas_x = padding + (x_value - min_x) * scale
        canvas_y = image_size - padding - (y_value - min_y) * scale
        return int(round(canvas_x)), int(round(canvas_y))

    return transform


def render_nas_mesh(
    nodes: dict[int, tuple[float, float]],
    triangles: list[tuple[int, int, int]],
    quads: list[tuple[int, int, int, int]],
    image_size: int = 1600,
    padding: int = 80,
    show_node_ids: bool = True,
) -> np.ndarray:
    """Draw a clean white-background rendering of the exported mesh."""
    canvas = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
    transform = _to_canvas_transform(nodes, image_size, padding)

    def polygon_points(node_ids: tuple[int, ...]) -> np.ndarray:
        return np.array([transform(nodes[node_id]) for node_id in node_ids], dtype=np.int32)

    for quad in quads:
        pts = polygon_points(quad)
        cv2.fillPoly(canvas, [pts], color=(220, 235, 255))
        cv2.polylines(canvas, [pts], isClosed=True, color=(40, 90, 180), thickness=2)

    for tri in triangles:
        pts = polygon_points(tri)
        cv2.fillPoly(canvas, [pts], color=(220, 255, 225))
        cv2.polylines(canvas, [pts], isClosed=True, color=(50, 150, 80), thickness=2)

    for node_id, point_xy in nodes.items():
        x_px, y_px = transform(point_xy)
        cv2.circle(canvas, (x_px, y_px), 4, (0, 0, 220), -1)
        if show_node_ids:
            cv2.putText(
                canvas,
                str(node_id),
                (x_px + 6, y_px - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

    return canvas


def render_nas_overlay(
    image: np.ndarray,
    nodes: dict[int, tuple[float, float]],
    triangles: list[tuple[int, int, int]],
    quads: list[tuple[int, int, int, int]],
    origin_point: tuple[int, int] | tuple[float, float],
    x_scale: float,
    y_scale: float,
    alpha: float = 0.45,
    show_node_ids: bool = True,
) -> np.ndarray:
    """Project exported model coordinates back onto the original source image."""
    overlay = image.copy()
    origin_col, origin_row = origin_point

    def to_image_point(point_xy: tuple[float, float]) -> tuple[int, int]:
        x_value, y_value = point_xy
        col = float(origin_col) + float(x_value) / float(x_scale)
        row = float(origin_row) - float(y_value) / float(y_scale)
        return int(round(col)), int(round(row))

    def polygon_points(node_ids: tuple[int, ...]) -> np.ndarray:
        return np.array([to_image_point(nodes[node_id]) for node_id in node_ids], dtype=np.int32)

    for quad in quads:
        pts = polygon_points(quad)
        cv2.fillPoly(overlay, [pts], color=(220, 235, 255))
        cv2.polylines(overlay, [pts], isClosed=True, color=(40, 90, 180), thickness=2)

    for tri in triangles:
        pts = polygon_points(tri)
        cv2.fillPoly(overlay, [pts], color=(220, 255, 225))
        cv2.polylines(overlay, [pts], isClosed=True, color=(50, 150, 80), thickness=2)

    blended = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0)

    for node_id, point_xy in nodes.items():
        col, row = to_image_point(point_xy)
        cv2.circle(blended, (col, row), 3, (0, 0, 220), -1)
        if show_node_ids:
            cv2.putText(
                blended,
                str(node_id),
                (col + 5, row - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

    return blended


def save_nas_mesh_preview(
    nas_path: str | Path,
    output_path: str | Path,
    image_size: int = 1600,
    show_node_ids: bool = True,
) -> Path:
    """Render and save the clean preview in one helper call."""
    nas_path = Path(nas_path)
    output_path = Path(output_path)
    nodes, triangles, quads = parse_nas_file(nas_path)
    preview = render_nas_mesh(
        nodes,
        triangles,
        quads,
        image_size=image_size,
        show_node_ids=show_node_ids,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), preview):
        raise ValueError(f"Could not write preview image: {output_path}")
    return output_path


def save_nas_overlay_preview(
    nas_path: str | Path,
    image_path: str | Path,
    output_path: str | Path,
    origin_point: tuple[int, int] | tuple[float, float],
    x_scale: float,
    y_scale: float,
    alpha: float = 0.45,
    show_node_ids: bool = True,
) -> Path:
    """Render and save the source-image overlay in one helper call."""
    nas_path = Path(nas_path)
    image_path = Path(image_path)
    output_path = Path(output_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    nodes, triangles, quads = parse_nas_file(nas_path)
    preview = render_nas_overlay(
        image,
        nodes,
        triangles,
        quads,
        origin_point=origin_point,
        x_scale=x_scale,
        y_scale=y_scale,
        alpha=alpha,
        show_node_ids=show_node_ids,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), preview):
        raise ValueError(f"Could not write overlay image: {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """CLI for one-off rendering of a single NAS file."""
    parser = argparse.ArgumentParser(
        description="Render a NAS mesh file to an image preview.",
    )
    parser.add_argument("nas_path", type=Path, help="Path to the NAS file to visualize.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path. Defaults next to the NAS file.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1600,
        help="Square output image size in pixels.",
    )
    parser.add_argument(
        "--hide-node-ids",
        action="store_true",
        help="Do not draw node id labels.",
    )
    return parser


def main() -> None:
    """Default command-line behavior: render one clean NAS preview image."""
    args = build_parser().parse_args()
    output_path = args.output or args.nas_path.with_suffix(".png")

    nodes, triangles, quads = parse_nas_file(args.nas_path)
    output_path = save_nas_mesh_preview(
        args.nas_path,
        output_path,
        image_size=args.image_size,
        show_node_ids=not args.hide_node_ids,
    )

    print(
        f"Saved preview to {output_path} "
        f"with {len(nodes)} nodes, {len(quads)} quads, {len(triangles)} triangles."
    )


if __name__ == "__main__":
    main()
