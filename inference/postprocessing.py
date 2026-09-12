import numpy as np


def create_binary_mask(prediction, threshold=0.5):
    """
    Convert U-Net probability map into a binary oil-spill mask.
    """

    probability_map = prediction.squeeze()

    binary_mask = (probability_map >= threshold).astype(np.uint8)

    return binary_mask

def calculate_spill_area(binary_mask):
    """
    Calculate the predicted oil-spill area in pixels
    and percentage of the image covered by oil.
    """

    oil_pixels = np.sum(binary_mask)
    total_pixels = binary_mask.size

    area_percentage = (oil_pixels / total_pixels) * 100

    return {
        "oil_pixels": int(oil_pixels),
        "area_percentage": float(area_percentage)
    }

def calculate_centroid(binary_mask):
    """
    Calculate the centroid (center point) of the detected oil spill.
    """

    y_coords, x_coords = np.where(binary_mask == 1)

    if len(x_coords) == 0:
        return {
            "centroid_x": None,
            "centroid_y": None
        }

    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)

    return {
        "centroid_x": float(centroid_x),
        "centroid_y": float(centroid_y)
    }

def calculate_bounding_box(binary_mask):
    """
    Calculate the bounding box of the detected oil spill.

    Returns:
        x_min, y_min, x_max, y_max
    """

    y_coords, x_coords = np.where(binary_mask == 1)

    if len(x_coords) == 0:
        return {
            "x_min": None,
            "y_min": None,
            "x_max": None,
            "y_max": None
        }

    return {
        "x_min": int(np.min(x_coords)),
        "y_min": int(np.min(y_coords)),
        "x_max": int(np.max(x_coords)),
        "y_max": int(np.max(y_coords))
    }

def calculate_perimeter(binary_mask):
    """
    Calculate the perimeter of the detected oil-spill region.
    """

    # Pad the mask so boundaries are handled correctly
    padded = np.pad(binary_mask, 1, mode="constant")

    perimeter = 0

    for y in range(1, padded.shape[0] - 1):
        for x in range(1, padded.shape[1] - 1):

            if padded[y, x] == 1:

                # Check 4-connected neighbors
                if padded[y - 1, x] == 0:
                    perimeter += 1

                if padded[y + 1, x] == 0:
                    perimeter += 1

                if padded[y, x - 1] == 0:
                    perimeter += 1

                if padded[y, x + 1] == 0:
                    perimeter += 1

    return float(perimeter)


def create_spill_result(
    oil_probability,
    spill_area,
    centroid,
    bounding_box,
    perimeter
):
    """
    Create a structured MARIS oil-spill detection result.
    """

    return {
        "oil_probability": float(oil_probability),

        "spill_area": {
            "oil_pixels": spill_area["oil_pixels"],
            "area_percentage": spill_area["area_percentage"]
        },

        "centroid": {
            "x": centroid["centroid_x"],
            "y": centroid["centroid_y"]
        },

        "bounding_box": bounding_box,

        "perimeter": float(perimeter)
    }