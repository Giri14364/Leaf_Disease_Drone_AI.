import numpy as np
from PIL import Image, ImageFilter


def check_image_quality(image):

    warnings = []

    # Convert image to RGB
    image = image.convert("RGB")

    # Check image dimensions
    width, height = image.size

    if width < 200 or height < 200:

        warnings.append(
            "Image resolution is too low. "
            "Please upload a larger image."
        )


    # Convert image to grayscale
    grayscale_image = image.convert(
        "L"
    )

    image_array = np.array(
        grayscale_image
    )


    # Check brightness
    average_brightness = (
        np.mean(image_array)
    )

    if average_brightness < 40:

        warnings.append(
            "Image is too dark. "
            "Please use a well-lit image."
        )

    elif average_brightness > 240:

        warnings.append(
            "Image is too bright. "
            "Please avoid excessive lighting."
        )


    # Check blur
    blurred_image = grayscale_image.filter(
        ImageFilter.GaussianBlur(
            radius=2
        )
    )

    original_array = np.array(
        grayscale_image,
        dtype=np.float32
    )

    blurred_array = np.array(
        blurred_image,
        dtype=np.float32
    )

    difference = (
        original_array - blurred_array
    )

    blur_score = np.var(
        difference
    )


    if blur_score < 10:

        warnings.append(
            "Image may be blurry. "
            "Please upload a clearer leaf image."
        )


    # Final result
    if warnings:

        return False, warnings

    return True, []