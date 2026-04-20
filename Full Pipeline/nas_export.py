from __future__ import annotations

"""Convert the final recovered graph faces into competition NAS bulk-data files."""

from pathlib import Path
import math


def file_ref_from_path(image_path: str | Path) -> int:
    """Extract the integer file reference from names like hole_042_normalised.jpg."""
    path = Path(image_path)
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not infer file reference from {path.name}")
    return int(digits)


def _format_float(value: float) -> str:
    return f"{value:.5f}"


def _grid_line(node_id: int, x_value: float, y_value: float) -> str:
    return (
        f"GRID      {node_id:>6}"
        f"      {_format_float(x_value):>10}"
        f"      {_format_float(y_value):>10}"
        f"      0."
    )


def _tria_line(element_id: int, file_ref: int, node_ids: tuple[int, int, int]) -> str:
    return (
        f"CTRIA3    {element_id:>6}      {file_ref:>3}"
        f"      {node_ids[0]:>6}      {node_ids[1]:>6}      {node_ids[2]:>6}"
    )


def _quad_line(element_id: int, file_ref: int, node_ids: tuple[int, int, int, int]) -> str:
    return (
        f"CQUAD4    {element_id:>6}      {file_ref:>3}"
        f"      {node_ids[0]:>6}      {node_ids[1]:>6}      {node_ids[2]:>6}      {node_ids[3]:>6}"
    )


def _polygon_area(coords_xy: list[tuple[float, float]]) -> float:
    """Signed area used to normalize triangle/quad winding before export."""
    area = 0.0
    for idx, (x1, y1) in enumerate(coords_xy):
        x2, y2 = coords_xy[(idx + 1) % len(coords_xy)]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def _rotate_to_smallest(values: list[int]) -> list[int]:
    smallest = min(values)
    start = values.index(smallest)
    return values[start:] + values[:start]


def _normalize_cycle(
    node_ids: list[int],
    original_node_ids: list[int],
    node_coords_xy: dict[int, tuple[float, float]],
) -> list[int]:
    coords_xy = [node_coords_xy[node_id] for node_id in original_node_ids]
    if _polygon_area(coords_xy) < 0:
        node_ids = [node_ids[0]] + list(reversed(node_ids[1:]))
    return _rotate_to_smallest(node_ids)


def calibrate_nodes_to_model(
    node_map: dict[int, tuple[int, int]],
    origin_point: tuple[int, int] | tuple[float, float],
    x_scale: float,
    y_scale: float,
) -> dict[int, tuple[float, float]]:
    """Map image-space nodes from (row, col) into model-space (x, y)."""
    origin_col, origin_row = origin_point

    if not math.isfinite(float(x_scale)) or not math.isfinite(float(y_scale)):
        raise ValueError(f"Invalid axis scales: x_scale={x_scale}, y_scale={y_scale}")

    nodes_model: dict[int, tuple[float, float]] = {}
    for node_id, (row, col) in node_map.items():
        x_value = (int(col) - float(origin_col)) * float(x_scale)
        y_value = (float(origin_row) - int(row)) * float(y_scale)
        nodes_model[int(node_id)] = (x_value, y_value)

    return nodes_model


def _collect_used_node_ids(
    triangles: list[list[int] | tuple[int, int, int]],
    quads: list[list[int] | tuple[int, int, int, int]],
) -> list[int]:
    """Keep only nodes referenced by exported faces unless told otherwise."""
    used: set[int] = set()
    for face in quads:
        used.update(int(node_id) for node_id in face)
    for face in triangles:
        used.update(int(node_id) for node_id in face)
    return sorted(used)


def write_nas_from_pipeline(
    node_map: dict[int, tuple[int, int]],
    triangles: list[list[int] | tuple[int, int, int]],
    quads: list[list[int] | tuple[int, int, int, int]],
    image_path: str | Path,
    origin_point: tuple[int, int] | tuple[float, float],
    x_scale: float,
    y_scale: float,
    output_path: str | Path,
    include_unused_nodes: bool = False,
) -> Path:
    """Write a NAS file directly from the final Full Pipeline graph outputs."""
    image_path = Path(image_path)
    output_path = Path(output_path)

    nodes_model_by_original_id = calibrate_nodes_to_model(node_map, origin_point, x_scale, y_scale)

    if include_unused_nodes:
        original_node_ids = sorted(int(node_id) for node_id in node_map)
    else:
        original_node_ids = _collect_used_node_ids(triangles, quads)

    missing = [node_id for node_id in original_node_ids if node_id not in node_map]
    if missing:
        raise ValueError(f"Faces reference missing node ids: {missing}")

    old_to_new = {
        original_id: new_id
        for new_id, original_id in enumerate(original_node_ids, start=1)
    }

    lines = ["BEGIN BULK"]
    for original_id in original_node_ids:
        x_value, y_value = nodes_model_by_original_id[original_id]
        lines.append(_grid_line(old_to_new[original_id], x_value, y_value))

    file_ref = file_ref_from_path(image_path)
    element_id = 1

    for face in quads:
        original_face = [int(node_id) for node_id in face]
        remapped_face = [old_to_new[node_id] for node_id in original_face]
        remapped_face = _normalize_cycle(remapped_face, original_face, nodes_model_by_original_id)
        lines.append(_quad_line(element_id, file_ref, tuple(remapped_face)))
        element_id += 1

    for face in triangles:
        original_face = [int(node_id) for node_id in face]
        remapped_face = [old_to_new[node_id] for node_id in original_face]
        remapped_face = _normalize_cycle(remapped_face, original_face, nodes_model_by_original_id)
        lines.append(_tria_line(element_id, file_ref, tuple(remapped_face)))
        element_id += 1

    lines.append("ENDDATA")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
