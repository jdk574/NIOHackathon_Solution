from further_preprocessing import mask_circles, remove_outside_axes
from skimage.morphology import skeletonize
from utils import *
from visualization import *
from further_preprocessing import *
from axes_extraction import *
from circles_det_conv import *
from blob_reconnection import *

"""Image preprocessing and skeleton extraction used before graph construction."""



image_path = "../jpeg images/hole_071_normalised.jpg"
image = open_image(image_path)

def preprocess(binary, kernel=(2,2), iterations=3):
    """Simple dilation used to reconnect thin gaps before skeletonization."""
    kernel = np.ones(kernel, np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=iterations)
    return dilated


def preprocess_and_skeletonize(image, work_area=None, dilation=True):
    """Turn the original JPEG into a cleaned binary skeleton ready for tracing."""

    img = open_image(image)

    img = turn_green_to_black(img)

    # Detect repeated small circles before the graph-building stages. These
    # artifacts can break skeleton connectivity if left untouched.
    centers, template, response = find_circles(img)

    # Use the detected axes to limit processing to the actual work area.
    axes_result = run_axes(image_path)

    work_area = axes_result["work_area"]
    vertical_axis = axes_result["vertical_axis"]

    image, axes_mask = remove_axes(image, remove=False)


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = remove_axes_label(gray, vertical_axis)
    gray = remove_mask(gray, axes_mask)

    blur = cv2.GaussianBlur(gray, (5, 5), 0.5)

    binary0 = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        3, 6
    )

    binary = remove_outside_axes(binary0, work_area)

    if dilation:
        dilated = preprocess(binary)
    else:
        dilated = binary.copy()

    skeleton = skeletonize(dilated // 255)
    skeleton = skeleton.astype(np.uint8)

    if centers is not None:
        masked, centers = mask_circles_yx_enhanced(skeleton, centers)
    else:
        masked = skeleton

    repaired, debug = connect_all_blob_centers(
        masked,
        centers,
        outer_radius=20,
        inner_radius=8,
        min_branch_length=3,
        connect_from="closest_pixel",
        min_path_length=3,
        max_connection_gap=20,
        require_near_inner_boundary=False,
        return_debug=True,
    )

    skeleton = repaired.copy()
    skeleton = skeleton // 255
    skeleton = skeleton.astype(np.uint8)

    return img, skeleton
