def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Do not call mobilenet_v2.preprocess_input()
    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    top_indices = np.argsort(
        predictions
    )[::-1][:5]

    results = []

    for index in top_indices:
        results.append({
            "class": class_names[index],
            "confidence": float(predictions[index]) * 100
        })

    return results