import tensorflow as tf


# ==============================
# Classification Preprocessing
# ==============================

CLASSIFIER_IMG_SIZE = (224, 224)


def preprocess_for_classifier(image_path):
    """
    Load and preprocess a SAR image for the oil-spill classifier.

    Training preprocessing:
    - Resize to 224x224
    - RGB, 3 channels
    - Normalize pixel values to [0, 1]
    """

    image = tf.keras.utils.load_img(
        image_path,
        target_size=CLASSIFIER_IMG_SIZE,
        color_mode="rgb"
    )

    image = tf.keras.utils.img_to_array(image)

    # Same normalization used during training
    image = image / 255.0

    # Add batch dimension
    image = tf.expand_dims(image, axis=0)

    return image


# ==============================
# Segmentation Preprocessing
# ==============================

SEGMENTATION_IMG_SIZE = (256, 256)


def preprocess_for_segmentation(image_path):
    """
    Load and preprocess a SAR image for the U-Net segmentation model.

    Training preprocessing:
    - Resize to 256x256
    - RGB, 3 channels
    - Normalize pixel values to [0, 1]
    """

    image = tf.keras.utils.load_img(
        image_path,
        target_size=SEGMENTATION_IMG_SIZE,
        color_mode="rgb"
    )

    image = tf.keras.utils.img_to_array(image)

    # Same normalization used during segmentation training
    image = image / 255.0

    # Add batch dimension
    image = tf.expand_dims(image, axis=0)

    return image