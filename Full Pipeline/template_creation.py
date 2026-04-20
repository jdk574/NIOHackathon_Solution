"""Offline helper showing how the saved circle template was originally built."""

from edge_detection import build_graph_from_skeleton
from node_extraction import get_skeleton_nodes
from further_preprocessing import extract_circular_patch_subpixel
from circles_det_conv import find_circles
from visualization import visualize_graph
from utils import open_image


def run_edge_detection(image, work_area=None):
    """Debug helper used only while sampling template patches from known circles."""
    img = open_image(image)

    skeleton, nodes, node_map, cl_node_map = get_skeleton_nodes(img, work_area)

    graph = build_graph_from_skeleton(skeleton, nodes)

    vis = visualize_graph(
        skeleton,
        graph["nodes"],
        graph["edge_paths"],
        scale=1,
        show_ids=True
    )

    print(type(graph["nodes"]))
    print(len(graph["nodes"]))

    return {
        "skeleton": skeleton,
        "graph": graph,
        "visualization": vis,
        "node_map": node_map
    }

image_path = "../University-Data-Challenge-main/jpeg images/hole_099_normalised.jpg"
img = cv2.imread(image_path)

result = run_edge_detection(image_path)
vis = result["visualization"]
cv2.imshow("Visualization", vis)

indices = (56,38,7,20)
coor = []


patches = []
print(type(patches))

y, x = result["node_map"][17]
x += 4.
y -= 4.
patch = extract_circular_patch_subpixel(image_path, x, y, radius=4)
coor.append((x,y))
patches.append(patch)



y, x = result["node_map"][38]
x += 3.5
y += 3.
patch = extract_circular_patch_subpixel(image_path, x, y, radius=4)
coor.append((x,y))
patches.append(patch)



y, x = result["node_map"][7]
x += 2.
y += 5.
patch = extract_circular_patch_subpixel(image_path, x, y, radius=4)
coor.append((x,y))
patches.append(patch)


y, x = result["node_map"][20]
x += 0.
y += 4.5
patch = extract_circular_patch_subpixel(image_path, x, y, radius=4)
coor.append((x,y))
patches.append(patch)



centers, template, response = find_circles(image_path, patches)
print(centers)
