# NIOHackathon Solution

Python solution for the NIO FE-mesh reconstruction challenge. The submission entrypoint is [`main.py`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/main.py), which delegates to the self-contained pipeline in [`Full Pipeline`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/Full%20Pipeline:1).

## Pipeline Summary

The current pipeline processes each competition image as follows:

1. Detect the magenta axes and recover the image-to-model coordinate transform.
2. Detect repeated small circular marks by normalized template matching and suppress them before graph extraction.
3. Threshold and skeletonize the mesh drawing.
4. Detect candidate mesh nodes from skeleton endpoints and junctions.
5. Classify and filter valid nodes.
6. Trace skeleton edges to build a graph.
7. Recover residual edges and split edges at strong turns.
8. Extract mesh faces and classify them into `CQUAD4` and `CTRIA3`.
9. Export the final mesh to Nastran bulk-data `.nas`.
10. Save both a clean mesh preview and an overlay preview on the source JPEG.

## Repository Layout

- [`main.py`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/main.py): submission entrypoint required by the brief
- [`Full Pipeline`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/Full%20Pipeline:1): full mesh-reconstruction implementation
- [`jpeg images`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/jpeg%20images:1): competition JPEG inputs
- [`submission_outputs_full_pipeline`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/submission_outputs_full_pipeline:1): generated `.nas` files and previews
- [`validate_submission.py`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/validate_submission.py:1): structural validator for generated NAS files

## Full Pipeline File Guide

- [`Full Pipeline/iter_pipeline.py`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/Full%20Pipeline/iter_pipeline.py:1): main batch runner that loops over all JPEGs, runs the full reconstruction pipeline, writes `.nas` files, and saves previews without stopping on a single-image failure
- [`Full Pipeline/pipeline.py`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/Full%20Pipeline/pipeline.py:1): single-image interactive/debug entrypoint showing the whole pipeline step by step on one example image
- [`Full Pipeline/preprocessing.py`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/Full%20Pipeline/preprocessing.py:1): turns the input image into a cleaned binary skeleton by masking axes, suppressing circles, thresholding, and skeletonizing
- [`Full Pipeline/axes_extraction.py`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/Full%20Pipeline/axes_extraction.py:1): detects the magenta axes, finds their intersection, and computes the pixel-to-model scaling used by export and overlays
- [`Full Pipeline/circles_det_conv.py`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/Full%20Pipeline/circles_det_conv.py:1): detects repeated circular artifacts with template matching, non-max suppression, and subpixel peak refinement
- [`Full Pipeline/template_creation.py`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/Full%20Pipeline/template_creation.py:1): offline helper showing how the saved circle template was built from sampled patches; included for documentation, not used by the runtime submission flow
- [`Full Pipeline/further_preprocessing.py`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/Full%20Pipeline/further_preprocessing.py:1): utility functions for removing masks, axis labels, and circle regions, plus small image-repair helpers used during cleanup
- [`Full Pipeline/blob_reconnection.py`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/Full%20Pipeline/blob_reconnection.py:1): reconnects local branch fragments around removed circles so broken skeleton segments can join back into the mesh
- [`Full Pipeline/node_extraction.py`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/Full%20Pipeline/node_extraction.py:1): finds raw node candidates from skeleton endpoints and junctions, traces open ends, and clusters nearby detections into stable node locations
- [`Full Pipeline/nodes_classification.py`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/Full%20Pipeline/nodes_classification.py:1): validates candidate nodes by inspecting the directions of nearby skeleton branches and rejecting points that behave like plain line pixels instead of true graph nodes
- [`Full Pipeline/edge_detection.py`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/Full%20Pipeline/edge_detection.py:1): traces skeleton paths between accepted nodes and builds the first graph representation of nodes, edges, edge paths, and adjacency
- [`Full Pipeline/extract_residual_edges.py`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/Full%20Pipeline/extract_residual_edges.py:1): removes already-explained edges from the skeleton, searches the leftover structure for missed nodes and edges, and merges that residual graph back into the main graph
- [`Full Pipeline/global_turns_detection.py`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/Full%20Pipeline/global_turns_detection.py:1): detects strong direction changes along traced edge paths and splits those paths at hidden corners so the graph matches the polygon structure better
- [`Full Pipeline/face_extraction.py`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/Full%20Pipeline/face_extraction.py:1): walks the recovered graph to extract closed faces and keeps only interior triangles and convex quads for export
- [`Full Pipeline/nas_export.py`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/Full%20Pipeline/nas_export.py:1): converts final nodes, triangles, and quads into the competition `.nas` format using the detected origin and scaling convention
- [`Full Pipeline/nas_visualize.py`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/Full%20Pipeline/nas_visualize.py:1): reads a `.nas` file and renders either a clean mesh preview or an overlay preview back onto the original image
- [`Full Pipeline/iter_nas_visualize.py`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/Full%20Pipeline/iter_nas_visualize.py:1): batch utility that regenerates clean and overlay previews for all existing `.nas` outputs
- [`Full Pipeline/extract_center_line.py`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/Full%20Pipeline/extract_center_line.py:1): contains the custom kernel rule used during axis-mask cleanup before Hough-based axis detection
- [`Full Pipeline/visualization.py`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/Full%20Pipeline/visualization.py:1): debug visualization helpers for skeletons, detected nodes, traced graphs, circles, and extracted faces
- [`Full Pipeline/utils.py`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/Full%20Pipeline/utils.py:1): shared low-level helpers for loading images, HSV masking, node-map filtering, and small image transforms

## Setup

Install the dependencies listed in [`requirements.txt`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/requirements.txt:1):

```bash
python3 -m pip install -r requirements.txt
```

If you use a Conda environment, activate it first and then run the same command.

## Run The Submission Pipeline

Run the full competition batch with:

```bash
python3 main.py
```

By default this reads from:

- [`jpeg images`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/jpeg%20images:1)

and writes to:

- [`submission_outputs_full_pipeline`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/submission_outputs_full_pipeline:1)

The batch runner continues even if one image fails. Any failures are recorded in:

- [`submission_outputs_full_pipeline/failed_images.txt`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/submission_outputs_full_pipeline/failed_images.txt:1)

## Useful Flags

```bash
python3 main.py --limit 5
python3 main.py --output-dir custom_outputs
python3 main.py --input-dir "jpeg images"
```

- `--limit N`: process only the first `N` images
- `--output-dir PATH`: choose where `.nas` files and previews are written
- `--input-dir PATH`: point to an alternate JPEG directory

## Outputs

For each successful image, the pipeline writes:

- `submission_outputs_full_pipeline/<image_stem>.nas`
- `submission_outputs_full_pipeline/previews/<image_stem>_mesh.png`
- `submission_outputs_full_pipeline/previews/<image_stem>_overlay.png`

## Circle Template Stage

Before the main skeleton graph is built, the pipeline runs a circle-template matching stage from [`circles_det_conv.py`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/Full%20Pipeline/circles_det_conv.py:1). The goal is to locate repeated small circular marks that can interfere with skeleton tracing.

The working idea is:

- use a precomputed normalized template stored at [`Full Pipeline/templates/template_99_in.npy`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/Full%20Pipeline/templates/template_99_in.npy:1)
- normalize the image/template comparison and run `cv2.matchTemplate(..., TM_CCOEFF_NORMED)` over the grayscale image
- keep only high-response peaks and apply a simple non-max suppression so the same circle is not detected many times
- refine peak locations to subpixel accuracy with a local quadratic fit
- convert the detected centers into preprocessing masks, where these circular structures can be removed or collapsed before skeleton graph extraction

In other words, this stage acts like a small convolution-style detector for repeated circular artifacts. It is not trying to solve the whole mesh problem; it simply cleans the image so later skeleton tracing sees the real line structure more clearly.

The file [`Full Pipeline/template_creation.py`](/Users/julykolev/PycharmProjects/NIOHackathon_Solution/Full%20Pipeline/template_creation.py:1) is included only to document how the template was originally constructed from sampled patches. It is not part of the submitted runtime pipeline executed by `main.py`.

## Validate NAS Files

```bash
python3 validate_submission.py submission_outputs_full_pipeline
```

## Regenerate Visualizations Only

If `.nas` files already exist and only the previews need to be rebuilt:

```bash
python3 "Full Pipeline/iter_nas_visualize.py"
```

## Notes

- `main.py` is the intended reproducible submission entrypoint.
- The exporter uses the numeric part of the filename as the file reference number.
- Coordinates are scaled into the competition's `100 x 75` frame using the detected axes.
- Circle template creation was an offline calibration step; runtime uses the saved template directly.
- The current approach is a deterministic computer-vision pipeline rather than a learned model.
