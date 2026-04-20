"""Single-image experimental entrypoint for stepping through the full pipeline."""

from pathlib import Path

from axes_extraction import *
from edge_detection import *
from face_extraction import *
from nas_export import write_nas_from_pipeline
from nas_visualize import save_nas_mesh_preview, save_nas_overlay_preview
from node_extraction import *
from preprocessing import *
from nodes_classification import *
from visualization import *
from extract_residual_edges import *
from global_turns_detection import *
from further_preprocessing import *
from utils import *




# This script is intentionally kept as an interactive sandbox for one image.
image_path = "../jpeg images/hole_100_normalised.jpg"
image = open_image(image_path)

# Detect the axes once up front so later export/overlay steps share the same origin.
axes_result = run_axes(image_path)

# Build the cleaned skeleton and the initial node candidates from it.
skeleton, nodes, node_map, cl_node_map = get_skeleton_nodes(
    image,
    trace_endpoints=True,
    trace_snap_radius=3,
    trace_max_extension=25,
    trace_search_radius=2,
)

# Validate the raw candidates so only stable junction-like points survive.
confirmed_nodes, rejected_nodes, all_results = classify_all_nodes(skeleton, nodes)
print(len(confirmed_nodes))
print(len(nodes))
print(rejected_nodes)


# First graph pass: trace edges directly between the confirmed nodes.
graph = build_graph_from_skeleton(skeleton, confirmed_nodes)



print(graph["edges"])

# Second graph pass: search the residual skeleton for edges or nodes the first pass missed.
recovered = recover_leftover_graph(
    skeleton=skeleton,
    old_nodes=graph["nodes"],
    old_edges=graph["edges"],
    old_edge_paths=graph["edge_paths"],
    match_radius=20,
    classify_candidates=False,
    old_node_mask_radius=6,   # new
    remove_end_neighborhood=0
)

new_edges = recovered["mapping_info"]["num_new_edges_added"]
print(f"Num of new edges added: {new_edges}")

subset_recovered = {k: recovered[k] for k in ["merged_nodes", "merged_edges", "merged_edge_paths"]}
subset_recovered["nodes"] = subset_recovered.pop("merged_nodes")
subset_recovered["edges"] = subset_recovered.pop("merged_edges")
subset_recovered["edge_paths"] = subset_recovered.pop("merged_edge_paths")

print(f'subset edges: {subset_recovered["edges"]}')

# Look for strong direction changes along each traced path so hidden corners become nodes.
curvature = check_edges_global_curvature(subset_recovered["edge_paths"],
                                         direction_window=12,
                                         comparison_step=6,
                                         angle_threshold_deg=13.0,
                                         min_peak_gap=10,
                                         ignore_ends=4
                                         )
print(curvature)
# Split edges at those turning points and rebuild the graph around the new nodes.
graph_new = split_edges_on_turns(subset_recovered,
                                 curvature,
                                 merge_dist=15,
                                 dbscan_eps=8,
                                 dbscan_min_samples=2,
                                 snap_path_endpoints=True
                                 )

print(f'first graph new: {graph_new["edges"]}')

print(recovered["merged_node_map"])
print(node_map)
print(graph["node_map"])

# Re-run node validation after splitting so only stable corners and junctions remain.
graph_new["nodes"], rejected_nodes, all_results = classify_all_nodes(skeleton, graph_new["nodes"],
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

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(graph_new["edges"])
# Convert the final graph cycles into triangle and quad mesh elements.
triangles, quads = extract_mesh_elements(graph_new["node_map"], graph_new["edges"])


visualize_faces(graph_new["node_map"], triangles, quads, image_path)

nas_output_path = Path("../submission_outputs_full_pipeline") / f"{Path(image_path).stem}.nas"
# Export the calibrated mesh to the challenge NAS format.
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

print(f"Wrote NAS output to {nas_output_path.resolve()}")

preview_dir = Path("../submission_outputs_full_pipeline/previews")
mesh_preview_path = preview_dir / f"{Path(image_path).stem}_mesh.png"
overlay_preview_path = preview_dir / f"{Path(image_path).stem}_overlay.png"

# Save both a clean mesh render and an image-space overlay for quick inspection.
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
print(f"Saved NAS mesh preview to {mesh_preview_path.resolve()}")
print(f"Saved NAS overlay preview to {overlay_preview_path.resolve()}")

