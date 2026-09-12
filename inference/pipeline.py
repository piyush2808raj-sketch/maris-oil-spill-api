from pathlib import Path
import json
import uuid

import numpy as np
from PIL import Image
import tensorflow as tf

from inference.preprocessing import (
    preprocess_for_classifier,
    preprocess_for_segmentation
)

from inference.postprocessing import (
    create_binary_mask,
    calculate_spill_area,
    calculate_centroid,
    calculate_bounding_box,
    calculate_perimeter,
    create_spill_result
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CLASSIFIER_MODEL_PATH = BASE_DIR / "ai_models" / "oil_spill_model.keras"
SEGMENTATION_MODEL_PATH = BASE_DIR / "ai_models" / "best_unet_oilspill.keras"

INPUT_DIR = BASE_DIR / "inference" / "input"
OUTPUT_DIR = BASE_DIR / "inference" / "output"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading models...")

classifier = tf.keras.models.load_model(
    CLASSIFIER_MODEL_PATH
)

segmentation_model = tf.keras.models.load_model(
    SEGMENTATION_MODEL_PATH,
    compile=False
)

print("Models loaded successfully.")


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_image(image_path):
    """
    Classifies the input image as oil or no oil.
    """

    image = preprocess_for_classifier(image_path)

    probability = classifier.predict(
        image,
        verbose=0
    )[0][0]

    probability = float(probability)

    if probability >= 0.5:

        label = "Class_1"
        result = "OIL_DETECTED"

    else:

        label = "Class_0"
        result = "NO_OIL"

    return {
        "class": label,
        "result": result,
        "oil_probability": probability
    }


# ============================================================
# SEGMENTATION
# ============================================================

def segment_image(image_path):
    """
    Generates the oil-spill probability mask.
    """

    image = preprocess_for_segmentation(image_path)

    prediction = segmentation_model.predict(
        image,
        verbose=0
    )[0]

    return prediction


# ============================================================
# SAVE BINARY MASK
# ============================================================

def save_binary_mask(binary_mask, output_path):

    mask_image = (
        binary_mask * 255
    ).astype(np.uint8)

    Image.fromarray(mask_image).save(
        output_path
    )


# ============================================================
# CREATE OVERLAY
# ============================================================

def create_overlay(
    image_path,
    binary_mask,
    output_path
):
    """
    Creates an image showing the detected
    oil-spill region over the original image.
    """

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (256, 256)
    )

    image_array = np.array(
        image
    ).astype(np.float32)

    mask = binary_mask.astype(bool)

    overlay = image_array.copy()

    # Highlight detected oil region
    overlay[mask] = (
        overlay[mask] * 0.4
        + np.array([255, 0, 0]) * 0.6
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

    Image.fromarray(
        overlay
    ).save(output_path)


# ============================================================
# SAVE JSON
# ============================================================

def save_result_json(
    result,
    output_path
):

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=4
        )


# ============================================================
# MAIN MARIS PIPELINE
# ============================================================

def run_pipeline(image_path):
    """
    Runs the complete MARIS inference pipeline.

    Returns:
        Dictionary containing:
        - job_id
        - classification
        - segmentation results
        - output file paths
    """

    image_path = Path(image_path)

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )


    # --------------------------------------------------------
    # CREATE UNIQUE JOB ID
    # --------------------------------------------------------

    job_id = uuid.uuid4().hex[:8]

    job_output_dir = (
        OUTPUT_DIR / job_id
    )

    job_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    classification_result = classify_image(
        image_path
    )


    # --------------------------------------------------------
    # OIL DETECTED
    # --------------------------------------------------------

    if classification_result["result"] == "OIL_DETECTED":

        print(
            "\nOil detected → Running segmentation..."
        )

        prediction = segment_image(
            image_path
        )

        binary_mask = create_binary_mask(
            prediction
        )


        # ----------------------------------------------------
        # POSTPROCESSING
        # ----------------------------------------------------

        spill_area = calculate_spill_area(
            binary_mask
        )

        centroid = calculate_centroid(
            binary_mask
        )

        bounding_box = calculate_bounding_box(
            binary_mask
        )

        perimeter = calculate_perimeter(
            binary_mask
        )


        # ----------------------------------------------------
        # CREATE RESULT
        # ----------------------------------------------------

        final_result = create_spill_result(

            oil_probability=
                classification_result[
                    "oil_probability"
                ],

            spill_area=
                spill_area,

            centroid=
                centroid,

            bounding_box=
                bounding_box,

            perimeter=
                perimeter
        )

        final_result["job_id"] = job_id

        final_result["result"] = (
            "OIL_DETECTED"
        )


        # ----------------------------------------------------
        # SAVE MASK
        # ----------------------------------------------------

        mask_path = (
            job_output_dir /
            "predicted_mask.png"
        )

        save_binary_mask(
            binary_mask,
            mask_path
        )


        # ----------------------------------------------------
        # SAVE OVERLAY
        # ----------------------------------------------------

        overlay_path = (
            job_output_dir /
            "overlay.png"
        )

        create_overlay(
            image_path,
            binary_mask,
            overlay_path
        )


        # ----------------------------------------------------
        # SAVE JSON
        # ----------------------------------------------------

        json_path = (
            job_output_dir /
            "result.json"
        )

        save_result_json(
            final_result,
            json_path
        )


    # --------------------------------------------------------
    # NO OIL
    # --------------------------------------------------------

    else:

        print(
            "\nNo oil detected → "
            "Skipping segmentation."
        )

        final_result = {

            "job_id": job_id,

            "result": "NO_OIL",

            "oil_probability":
                classification_result[
                    "oil_probability"
                ],

            "spill_area": None,

            "centroid": None,

            "bounding_box": None,

            "perimeter": None
        }


        # ----------------------------------------------------
        # SAVE JSON
        # ----------------------------------------------------

        json_path = (
            job_output_dir /
            "result.json"
        )

        save_result_json(
            final_result,
            json_path
        )

        mask_path = None
        overlay_path = None


    # ========================================================
    # RETURN API-FRIENDLY RESULT
    # ========================================================

    return {

        "job_id": job_id,

        "result": final_result,

        "files": {

            "mask":
                str(mask_path)
                if mask_path
                else None,

            "overlay":
                str(overlay_path)
                if overlay_path
                else None,

            "json":
                str(json_path)
        }
    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    image_path = (
        INPUT_DIR /
        "sample1.png"
    )

    print("\n===================================")
    print("       MARIS INFERENCE PIPELINE")
    print("===================================\n")

    output = run_pipeline(
        image_path
    )

    print("\nClassification / Detection Result:")
    print(output["result"])

    print("\nOutput files:")

    print(
        f"Mask    : {output['files']['mask']}"
    )

    print(
        f"Overlay : {output['files']['overlay']}"
    )

    print(
        f"JSON    : {output['files']['json']}"
    )

    print("\nJob ID:")
    print(output["job_id"])

    print("\n===================================")
    print("       PIPELINE COMPLETED")
    print("===================================")