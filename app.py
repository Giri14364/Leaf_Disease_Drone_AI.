
from pathlib import Path
from datetime import datetime
import csv
import io

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageStat
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Crop Disease Detection",
    page_icon="🌱",
    layout="wide",
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    PROJECT_DIR
    / "model"
    / "crop_disease_model_81classes.keras"
)

CLASS_NAMES_PATH = (
    PROJECT_DIR
    / "model"
    / "class_names_81.txt"
)

HISTORY_DIR = PROJECT_DIR / "history"

HISTORY_FILE = HISTORY_DIR / "prediction_history.csv"

HISTORY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)

TOP_K = 5

EXPECTED_CLASSES = 81


# ============================================================
# DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {
    "healthy": {
        "status": "Healthy",
        "solution": (
            "The leaf appears healthy. Continue regular irrigation, "
            "balanced fertilization, field monitoring and pest inspection."
        ),
        "prevention": (
            "Maintain proper spacing, irrigation, nutrition and regular "
            "crop monitoring."
        ),
    },

    "Apple scab": {
        "status": "Disease detected",
        "solution": (
            "Remove infected leaves and fallen plant material. Improve "
            "air circulation and consider an appropriate fungicide "
            "according to local agricultural recommendations."
        ),
        "prevention": (
            "Maintain orchard sanitation and avoid prolonged leaf wetness."
        ),
    },

    "Black rot": {
        "status": "Disease detected",
        "solution": (
            "Remove infected plant material and maintain orchard hygiene. "
            "Use recommended disease-control measures when required."
        ),
        "prevention": (
            "Remove diseased leaves and fruit and maintain good airflow."
        ),
    },

    "Cedar apple rust": {
        "status": "Disease detected",
        "solution": (
            "Remove severely affected material and follow locally "
            "recommended fungicide and orchard management practices."
        ),
        "prevention": (
            "Monitor the orchard regularly and maintain good sanitation."
        ),
    },

    "Leaf miner": {
        "status": "Pest detected",
        "solution": (
            "Remove heavily damaged leaves where practical and monitor "
            "for active leaf-miner infestation. Use integrated pest "
            "management methods."
        ),
        "prevention": (
            "Regular scouting and removal of severely affected leaves "
            "can reduce infestation."
        ),
    },

    "Red rust": {
        "status": "Disease detected",
        "solution": (
            "Remove affected material where practical and follow local "
            "agricultural recommendations for disease management."
        ),
        "prevention": (
            "Maintain field sanitation and adequate plant spacing."
        ),
    },

    "Brown spot": {
        "status": "Disease detected",
        "solution": (
            "Remove severely affected leaves where appropriate and "
            "maintain balanced crop nutrition and field sanitation."
        ),
        "prevention": (
            "Use healthy planting material and maintain good crop management."
        ),
    },

    "Mosaic": {
        "status": "Viral disease suspected",
        "solution": (
            "Remove severely infected plants where recommended and control "
            "possible insect vectors. Use healthy planting material."
        ),
        "prevention": (
            "Monitor vector populations and use disease-free planting material."
        ),
    },

    "Powdery mildew": {
        "status": "Fungal disease detected",
        "solution": (
            "Improve air circulation and reduce excessive humidity. "
            "Use an appropriate fungicide according to local recommendations."
        ),
        "prevention": (
            "Avoid excessive humidity and overcrowding."
        ),
    },

    "Bacterial spot": {
        "status": "Bacterial disease detected",
        "solution": (
            "Remove severely affected plant material and avoid overhead "
            "irrigation. Follow locally recommended bacterial disease "
            "management practices."
        ),
        "prevention": (
            "Use clean planting material and avoid unnecessary leaf wetness."
        ),
    },

    "Bacterial blight": {
        "status": "Bacterial disease detected",
        "solution": (
            "Remove severely affected material where practical and maintain "
            "field sanitation. Follow local agricultural recommendations."
        ),
        "prevention": (
            "Use healthy planting material and avoid spreading contaminated "
            "plant material."
        ),
    },

    "Bacterial canker": {
        "status": "Bacterial disease detected",
        "solution": (
            "Prune severely affected portions when appropriate and disinfect "
            "tools between plants. Follow local agricultural guidance."
        ),
        "prevention": (
            "Maintain sanitation and avoid moving infected plant material."
        ),
    },

    "Early blight": {
        "status": "Fungal disease detected",
        "solution": (
            "Remove affected leaves, improve airflow and avoid prolonged "
            "leaf wetness. Use recommended fungicide measures if necessary."
        ),
        "prevention": (
            "Use crop rotation, sanitation and good spacing."
        ),
    },

    "Late blight": {
        "status": "Fungal disease detected",
        "solution": (
            "Remove severely infected material and avoid overhead irrigation. "
            "Use locally recommended late-blight control measures."
        ),
        "prevention": (
            "Monitor crops frequently and maintain good field sanitation."
        ),
    },

    "Leaf blight": {
        "status": "Disease detected",
        "solution": (
            "Remove severely affected leaves and improve airflow. "
            "Use appropriate disease-management practices."
        ),
        "prevention": (
            "Maintain sanitation and avoid prolonged leaf wetness."
        ),
    },

    "Leaf spot": {
        "status": "Disease detected",
        "solution": (
            "Remove heavily affected leaves and improve air circulation. "
            "Use an appropriate fungicide if recommended locally."
        ),
        "prevention": (
            "Maintain crop spacing and reduce prolonged leaf wetness."
        ),
    },

    "Leaf scorch": {
        "status": "Disease or stress symptom detected",
        "solution": (
            "Inspect irrigation, nutrition and environmental conditions. "
            "Remove severely damaged leaves where appropriate."
        ),
        "prevention": (
            "Maintain balanced irrigation and nutrition."
        ),
    },

    "Leaf mold": {
        "status": "Fungal disease detected",
        "solution": (
            "Improve ventilation, reduce excessive humidity and remove "
            "severely affected leaves."
        ),
        "prevention": (
            "Maintain airflow and avoid excessive humidity."
        ),
    },

    "Septoria leaf spot": {
        "status": "Fungal disease detected",
        "solution": (
            "Remove affected leaves and plant debris and improve airflow. "
            "Follow locally recommended fungicide practices."
        ),
        "prevention": (
            "Use sanitation and avoid prolonged leaf wetness."
        ),
    },

    "Spider mites Two-spotted spider mite": {
        "status": "Pest detected",
        "solution": (
            "Inspect the underside of leaves and use integrated pest "
            "management. Follow locally approved control measures."
        ),
        "prevention": (
            "Regularly inspect leaves and avoid conditions that encourage "
            "severe mite infestation."
        ),
    },

    "Target Spot": {
        "status": "Fungal disease detected",
        "solution": (
            "Remove severely affected foliage and improve airflow. "
            "Use locally recommended disease-management practices."
        ),
        "prevention": (
            "Maintain sanitation and avoid prolonged leaf wetness."
        ),
    },

    "Tomato mosaic virus": {
        "status": "Viral disease suspected",
        "solution": (
            "Remove severely infected plants where recommended and prevent "
            "mechanical spread. Use healthy planting material."
        ),
        "prevention": (
            "Disinfect tools and use disease-free planting material."
        ),
    },

    "Tomato Yellow Leaf Curl Virus": {
        "status": "Viral disease suspected",
        "solution": (
            "Inspect and manage whitefly vectors using integrated pest "
            "management. Remove severely infected plants where appropriate."
        ),
        "prevention": (
            "Monitor whiteflies and use healthy planting material."
        ),
    },

    "Cercospora leaf spot Gray leaf spot": {
        "status": "Fungal disease detected",
        "solution": (
            "Remove severely affected material and maintain good airflow. "
            "Follow locally recommended fungicide practices."
        ),
        "prevention": (
            "Maintain sanitation and appropriate crop spacing."
        ),
    },

    "Common rust": {
        "status": "Fungal disease detected",
        "solution": (
            "Monitor disease progression and follow locally recommended "
            "fungicide and crop-management practices."
        ),
        "prevention": (
            "Use resistant varieties where available and monitor fields."
        ),
    },

    "Northern Leaf Blight": {
        "status": "Fungal disease detected",
        "solution": (
            "Remove heavily infected material where practical and use "
            "locally recommended disease-management practices."
        ),
        "prevention": (
            "Use resistant varieties and appropriate crop rotation."
        ),
    },

    "Streak virus": {
        "status": "Viral disease suspected",
        "solution": (
            "Remove severely affected plants where recommended and manage "
            "known insect vectors."
        ),
        "prevention": (
            "Use healthy planting material and monitor vector populations."
        ),
    },

    "Curl virus": {
        "status": "Viral disease suspected",
        "solution": (
            "Inspect for insect vectors and use integrated pest management. "
            "Remove severely affected plants where recommended."
        ),
        "prevention": (
            "Monitor and manage vector populations."
        ),
    },

    "Esca (Black Measles)": {
        "status": "Disease detected",
        "solution": (
            "Remove severely affected plant material where appropriate and "
            "follow local vineyard disease-management recommendations."
        ),
        "prevention": (
            "Maintain vineyard sanitation and monitor plants regularly."
        ),
    },

    "Anthracnose": {
        "status": "Fungal disease detected",
        "solution": (
            "Remove affected plant material and improve airflow. "
            "Follow locally recommended fungicide practices."
        ),
        "prevention": (
            "Maintain sanitation and avoid prolonged moisture on foliage."
        ),
    },

    "Cutting weevil": {
        "status": "Pest detected",
        "solution": (
            "Inspect plants for adult insects and feeding damage. "
            "Use integrated pest management according to local guidance."
        ),
        "prevention": (
            "Regular crop scouting can help detect infestations early."
        ),
    },

    "Die back": {
        "status": "Disease or stress symptom detected",
        "solution": (
            "Prune severely affected material where appropriate and inspect "
            "for fungal infection, pests, irrigation or nutritional stress."
        ),
        "prevention": (
            "Maintain good sanitation, nutrition and irrigation."
        ),
    },

    "Gall midge": {
        "status": "Pest detected",
        "solution": (
            "Inspect new growth and use integrated pest management "
            "according to local agricultural recommendations."
        ),
        "prevention": (
            "Regularly inspect young shoots and leaves."
        ),
    },

    "Sooty mould": {
        "status": "Fungal growth associated with insect honeydew",
        "solution": (
            "Control the underlying sap-sucking insect infestation and "
            "clean heavily affected foliage where practical."
        ),
        "prevention": (
            "Monitor and control aphids, scale insects and other sap feeders."
        ),
    },

    "Haunglongbing (Citrus greening)": {
        "status": "Bacterial disease suspected",
        "solution": (
            "Manage insect vectors and follow local citrus disease-control "
            "recommendations. Remove severely affected trees when advised."
        ),
        "prevention": (
            "Use healthy planting material and monitor vector populations."
        ),
    },

    "Ring spot": {
        "status": "Viral disease suspected",
        "solution": (
            "Remove severely affected plants where recommended and manage "
            "possible insect vectors."
        ),
        "prevention": (
            "Use healthy planting material and monitor crops regularly."
        ),
    },

    "Nutrition deficiency": {
        "status": "Nutritional stress detected",
        "solution": (
            "Check soil and crop nutrient conditions. Apply balanced "
            "fertilizer based on soil-test results and crop requirements."
        ),
        "prevention": (
            "Maintain balanced nutrition and perform periodic soil testing."
        ),
    },

    "White spot": {
        "status": "Disease or leaf symptom detected",
        "solution": (
            "Inspect affected plants closely and maintain sanitation. "
            "Follow locally recommended disease-management practices."
        ),
        "prevention": (
            "Maintain good crop spacing, sanitation and field monitoring."
        ),
    },

    "Grassy shoot": {
        "status": "Disease suspected",
        "solution": (
            "Remove severely affected plants where recommended and monitor "
            "for insect vectors."
        ),
        "prevention": (
            "Use healthy planting material and monitor vector populations."
        ),
    },

    "Caterpillar": {
        "status": "Pest detected",
        "solution": (
            "Inspect leaves for larvae and feeding damage. Use integrated "
            "pest management and locally approved controls if necessary."
        ),
        "prevention": (
            "Regular field scouting helps detect caterpillars early."
        ),
    },

    "Diabrotica speciosa": {
        "status": "Pest detected",
        "solution": (
            "Monitor for adult beetles and feeding damage and use integrated "
            "pest management according to local recommendations."
        ),
        "prevention": (
            "Regular monitoring and appropriate crop management are recommended."
        ),
    },
}


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None

if "last_image" not in st.session_state:
    st.session_state.last_image = None

if "last_mode" not in st.session_state:
    st.session_state.last_mode = "Upload Image"

if "drone_frame" not in st.session_state:
    st.session_state.drone_frame = None

if "drone_connected" not in st.session_state:
    st.session_state.drone_connected = False


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_data
def load_class_names():

    if not CLASS_NAMES_PATH.exists():

        raise FileNotFoundError(
            f"Class names file not found:\n{CLASS_NAMES_PATH}"
        )

    with CLASS_NAMES_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        names = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if len(names) != EXPECTED_CLASSES:

        raise ValueError(
            f"Expected {EXPECTED_CLASSES} classes, "
            f"but found {len(names)}."
        )

    return names


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    if model.output_shape[-1] != EXPECTED_CLASSES:

        raise ValueError(
            "The loaded model does not have "
            f"{EXPECTED_CLASSES} output classes."
        )

    return model


try:

    CLASS_NAMES = load_class_names()

    MODEL = load_model()

except Exception as error:

    st.error(
        "Unable to load the AI model or class names."
    )

    st.exception(error)

    st.stop()


# ============================================================
# FORMAT CLASS NAME
# ============================================================

def format_class_name(class_name):

    formatted = class_name

    formatted = formatted.replace(
        "___",
        " - ",
    )

    formatted = formatted.replace(
        "_",
        " ",
    )

    formatted = formatted.replace(
        "  ",
        " ",
    )

    return formatted.strip()


# ============================================================
# EXTRACT CROP AND DISEASE
# ============================================================

def extract_crop_and_disease(class_name):

    parts = class_name.split(
        "___",
        1,
    )

    if len(parts) == 2:

        crop = parts[0]
        disease = parts[1]

    else:

        crop = class_name
        disease = "Unknown"

    crop = crop.replace(
        "_",
        " ",
    )

    disease = disease.replace(
        "_",
        " ",
    )

    return crop, disease


# ============================================================
# DISEASE INFORMATION LOOKUP
# ============================================================

def get_disease_information(disease):

    normalized = disease.strip()

    if normalized.lower() == "healthy":

        return DISEASE_INFO["healthy"]

    for key, information in DISEASE_INFO.items():

        if key.lower() == normalized.lower():

            return information

    for key, information in DISEASE_INFO.items():

        if key.lower() in normalized.lower():

            return information

    return {
        "status": "Disease detected",

        "solution": (
            "Inspect the crop carefully and consult a local agricultural "
            "expert before applying any treatment."
        ),

        "prevention": (
            "Maintain good crop sanitation, irrigation, nutrition and "
            "regular disease monitoring."
        ),
    }


# ============================================================
# IMAGE QUALITY
# ============================================================

def check_image_quality(image):

    rgb_image = image.convert("RGB")

    width, height = rgb_image.size

    if width < 100 or height < 100:

        return (
            False,
            "Image resolution is too low. Please use a clearer image.",
        )

    grayscale = rgb_image.convert("L")

    statistics = ImageStat.Stat(grayscale)

    brightness = statistics.mean[0]

    variance = statistics.var[0]

    if brightness < 20:

        return (
            False,
            "Image is too dark. Please provide a brighter image.",
        )

    if brightness > 245:

        return (
            False,
            "Image is too bright. Please use a normally exposed image.",
        )

    if variance < 20:

        return (
            False,
            "Image appears too blurry or lacks visual detail.",
        )

    return (
        True,
        "Image quality is acceptable.",
    )


# ============================================================
# DRONE CAMERA CONNECTION
# ============================================================

def connect_to_drone(drone_url):

    if not drone_url:

        return None, "Please enter the drone camera URL."

    drone_url = drone_url.strip()

    try:

        camera = cv2.VideoCapture(drone_url)

        camera.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1,
        )

        if not camera.isOpened():

            camera.release()

            return (
                None,
                "Unable to connect to the drone camera. "
                "Check the URL and network connection.",
            )

        success, frame = camera.read()

        camera.release()

        if not success or frame is None:

            return (
                None,
                "Connected to the URL, but no video frame was received.",
            )

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(
            frame_rgb
        ).convert("RGB")

        return (
            image,
            "Drone camera connected successfully.",
        )

    except Exception as error:

        return (
            None,
            f"Drone connection failed: {error}",
        )


# ============================================================
# PREDICTION
# ============================================================

def predict_disease(image):

    """
    The saved MobileNetV2 model already contains preprocessing
    layers internally.

    Therefore DO NOT call:

        mobilenet_v2.preprocess_input()

    The model receives normal 0-255 RGB pixel values.
    """

    rgb_image = image.convert("RGB")

    resized_image = rgb_image.resize(
        IMAGE_SIZE
    )

    image_array = np.array(
        resized_image,
        dtype=np.float32,
    )

    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    predictions = MODEL.predict(
        image_array,
        verbose=0,
    )[0]

    top_indices = np.argsort(
        predictions
    )[::-1][:TOP_K]

    results = []

    for index in top_indices:

        class_name = CLASS_NAMES[index]

        confidence = float(
            predictions[index]
        ) * 100.0

        crop, disease = extract_crop_and_disease(
            class_name
        )

        information = get_disease_information(
            disease
        )

        results.append(
            {
                "class_name": class_name,

                "display_name": format_class_name(
                    class_name
                ),

                "crop": crop,

                "disease": disease,

                "confidence": confidence,

                "status": information["status"],

                "solution": information["solution"],

                "prevention": information["prevention"],
            }
        )

    return results


# ============================================================
# SAVE PREDICTION HISTORY
# ============================================================

def save_prediction_history(
    results,
    mode,
    latitude,
    longitude,
):

    if not results:

        return

    top_result = results[0]

    file_exists = HISTORY_FILE.exists()

    with HISTORY_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        if not file_exists:

            writer.writerow(
                [
                    "Date Time",
                    "Mode",
                    "Crop",
                    "Disease",
                    "Confidence",
                    "Latitude",
                    "Longitude",
                ]
            )

        writer.writerow(
            [
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

                mode,

                top_result["crop"],

                top_result["disease"],

                f"{top_result['confidence']:.2f}%",

                latitude,

                longitude,
            ]
        )


# ============================================================
# LOAD HISTORY
# ============================================================

def load_history():

    if not HISTORY_FILE.exists():

        return pd.DataFrame()

    try:

        dataframe = pd.read_csv(
            HISTORY_FILE
        )

        return dataframe

    except Exception:

        return pd.DataFrame()


# ============================================================
# CREATE PDF REPORT
# ============================================================

def create_pdf_report(
    image,
    results,
    mode,
    latitude,
    longitude,
):

    buffer = io.BytesIO()

    pdf = canvas.Canvas(
        buffer,
        pagesize=A4,
    )

    page_width, page_height = A4

    left_margin = 45

    y = page_height - 50

    top_result = results[0]

    pdf.setFont(
        "Helvetica-Bold",
        18,
    )

    pdf.drawString(
        left_margin,
        y,
        "AI Crop Disease Detection Report",
    )

    y -= 30

    pdf.setFont(
        "Helvetica",
        10,
    )

    pdf.drawString(
        left_margin,
        y,
        "Generated: "
        + datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    )

    y -= 18

    pdf.drawString(
        left_margin,
        y,
        f"Detection Mode: {mode}",
    )

    y -= 30

    pdf.setFont(
        "Helvetica-Bold",
        13,
    )

    pdf.drawString(
        left_margin,
        y,
        "Primary Detection",
    )

    y -= 22

    pdf.setFont(
        "Helvetica",
        11,
    )

    pdf.drawString(
        left_margin,
        y,
        f"Crop: {top_result['crop']}",
    )

    y -= 18

    pdf.drawString(
        left_margin,
        y,
        f"Disease: {top_result['disease']}",
    )

    y -= 18

    pdf.drawString(
        left_margin,
        y,
        f"Confidence: {top_result['confidence']:.2f}%",
    )

    y -= 18

    pdf.drawString(
        left_margin,
        y,
        f"Status: {top_result['status']}",
    )

    y -= 30

    pdf.setFont(
        "Helvetica-Bold",
        12,
    )

    pdf.drawString(
        left_margin,
        y,
        "Recommended Action",
    )

    y -= 20

    pdf.setFont(
        "Helvetica",
        9,
    )

    solution = top_result["solution"]

    solution_lines = []

    words = solution.split()

    current_line = ""

    for word in words:

        test_line = (
            current_line + " " + word
        ).strip()

        if len(test_line) > 90:

            solution_lines.append(
                current_line
            )

            current_line = word

        else:

            current_line = test_line

    if current_line:

        solution_lines.append(
            current_line
        )

    for line in solution_lines:

        pdf.drawString(
            left_margin,
            y,
            line,
        )

        y -= 14

    y -= 15

    pdf.setFont(
        "Helvetica-Bold",
        12,
    )

    pdf.drawString(
        left_margin,
        y,
        "Top 5 Predictions",
    )

    y -= 22

    pdf.setFont(
        "Helvetica",
        9,
    )

    for number, result in enumerate(
        results,
        start=1,
    ):

        text = (
            f"{number}. "
            f"{result['crop']} - "
            f"{result['disease']} - "
            f"{result['confidence']:.2f}%"
        )

        pdf.drawString(
            left_margin,
            y,
            text,
        )

        y -= 16

    y -= 15

    pdf.setFont(
        "Helvetica-Bold",
        12,
    )

    pdf.drawString(
        left_margin,
        y,
        "GPS Information",
    )

    y -= 20

    pdf.setFont(
        "Helvetica",
        10,
    )

    latitude_text = (
        str(latitude)
        if latitude
        else "Not provided"
    )

    longitude_text = (
        str(longitude)
        if longitude
        else "Not provided"
    )

    pdf.drawString(
        left_margin,
        y,
        f"Latitude: {latitude_text}",
    )

    y -= 16

    pdf.drawString(
        left_margin,
        y,
        f"Longitude: {longitude_text}",
    )

    y -= 30

    pdf.setFont(
        "Helvetica-Bold",
        10,
    )

    pdf.drawString(
        left_margin,
        y,
        "Important:",
    )

    y -= 15

    pdf.setFont(
        "Helvetica",
        8,
    )

    disclaimer = (
        "This AI result is intended as a decision-support tool. "
        "Confirm important crop disease decisions with a qualified "
        "agricultural professional."
    )

    disclaimer_lines = []

    words = disclaimer.split()

    current_line = ""

    for word in words:

        test_line = (
            current_line + " " + word
        ).strip()

        if len(test_line) > 100:

            disclaimer_lines.append(
                current_line
            )

            current_line = word

        else:

            current_line = test_line

    if current_line:

        disclaimer_lines.append(
            current_line
        )

    for line in disclaimer_lines:

        pdf.drawString(
            left_margin,
            y,
            line,
        )

        y -= 12

    pdf.save()

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🌱 AI Crop Disease Detection"
)

st.markdown(
    """
    **AI-Based Agricultural Crop Disease Detection System**

    Upload a crop leaf image, capture an image using your laptop
    camera, or connect to a drone camera using a camera stream URL
    to detect the crop, disease and confidence.
    """
)


# ============================================================
# SIDEBAR - IMAGE INPUT MODE
# ============================================================

with st.sidebar:

    st.header("📷 Image Input")

    input_mode = st.radio(
        "Select input mode:",
        [
            "📤 Upload Image",
            "📷 Laptop Camera",
            "🚁 Drone Mode",
        ],
    )

    st.divider()

    st.caption(
        "AI Model: MobileNetV2"
    )

    st.caption(
        "Classes: 81"
    )


# ============================================================
# IMAGE INPUT
# ============================================================

selected_image = None


# ============================================================
# UPLOAD IMAGE
# ============================================================

if input_mode == "📤 Upload Image":

    st.subheader(
        "📤 Upload Crop Leaf Image"
    )

    uploaded_file = st.file_uploader(
        "Upload crop leaf image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        help="Upload a clear crop leaf image.",
    )

    if uploaded_file is not None:

        try:

            selected_image = Image.open(
                uploaded_file
            ).convert("RGB")

        except Exception as error:

            st.error(
                "Unable to open the uploaded image."
            )

            st.exception(error)


# ============================================================
# LAPTOP CAMERA
# ============================================================

elif input_mode == "📷 Laptop Camera":

    st.subheader(
        "📷 Laptop Camera"
    )

    st.info(
        "Allow camera access in your browser."
    )

    camera_file = st.camera_input(
        "Take a picture of the crop leaf"
    )

    if camera_file is not None:

        try:

            selected_image = Image.open(
                camera_file
            ).convert("RGB")

        except Exception as error:

            st.error(
                "Unable to read the camera image."
            )

            st.exception(error)


# ============================================================
# DRONE MODE
# ============================================================

elif input_mode == "🚁 Drone Mode":

    st.subheader(
        "🚁 Drone Camera"
    )

    drone_input_type = st.radio(
        "Select drone input:",
        [
            "📁 Upload Drone Image",
            "🔗 Connect Drone Camera",
        ],
        horizontal=True,
    )

    # --------------------------------------------------------
    # DRONE IMAGE UPLOAD
    # --------------------------------------------------------

    if drone_input_type == "📁 Upload Drone Image":

        st.info(
            "Upload an image captured by the drone camera."
        )

        drone_file = st.file_uploader(
            "Upload drone camera image",
            type=[
                "jpg",
                "jpeg",
                "png",
            ],
            key="drone_upload",
        )

        if drone_file is not None:

            try:

                selected_image = Image.open(
                    drone_file
                ).convert("RGB")

            except Exception as error:

                st.error(
                    "Unable to open the drone image."
                )

                st.exception(error)

    # --------------------------------------------------------
    # DRONE CAMERA URL
    # --------------------------------------------------------

    else:

        st.info(
            "Enter the live camera stream URL provided by your "
            "drone or drone camera."
        )

        drone_url = st.text_input(
            "🔗 Drone Camera URL",
            placeholder=(
                "Example: rtsp://192.168.1.100:8554/live"
            ),
            help=(
                "Examples: RTSP or compatible HTTP camera stream URL."
            ),
        )

        st.caption(
            "Example RTSP URL: "
            "rtsp://192.168.1.100:8554/live"
        )

        connect_col1, connect_col2 = st.columns(2)

        with connect_col1:

            connect_button = st.button(
                "🔗 Connect to Drone",
                type="primary",
                use_container_width=True,
            )

        with connect_col2:

            capture_button = st.button(
                "📸 Capture Frame",
                use_container_width=True,
            )

        if connect_button:

            if not drone_url.strip():

                st.warning(
                    "Please enter the drone camera URL."
                )

            else:

                with st.spinner(
                    "Connecting to drone camera..."
                ):

                    drone_image, drone_message = (
                        connect_to_drone(
                            drone_url
                        )
                    )

                if drone_image is not None:

                    st.session_state.drone_frame = (
                        drone_image
                    )

                    st.session_state.drone_connected = (
                        True
                    )

                    st.success(
                        drone_message
                    )

                else:

                    st.session_state.drone_connected = (
                        False
                    )

                    st.error(
                        drone_message
                    )

        if capture_button:

            if not drone_url.strip():

                st.warning(
                    "Please enter the drone camera URL first."
                )

            else:

                with st.spinner(
                    "Capturing frame from drone..."
                ):

                    drone_image, drone_message = (
                        connect_to_drone(
                            drone_url
                        )
                    )

                if drone_image is not None:

                    st.session_state.drone_frame = (
                        drone_image
                    )

                    st.session_state.drone_connected = (
                        True
                    )

                    st.success(
                        "📸 Frame captured successfully."
                    )

                else:

                    st.error(
                        drone_message
                    )

        if st.session_state.drone_connected:

            st.success(
                "🟢 Drone camera connected"
            )

        if st.session_state.drone_frame is not None:

            st.subheader(
                "📡 Drone Camera Frame"
            )

            st.image(
                st.session_state.drone_frame,
                caption="Latest Drone Camera Frame",
                use_container_width=True,
            )

            selected_image = (
                st.session_state.drone_frame
            )


# ============================================================
# GPS INFORMATION
# ============================================================

st.subheader(
    "📍 GPS Information"
)

gps_col1, gps_col2 = st.columns(2)

with gps_col1:

    latitude = st.text_input(
        "Latitude",
        placeholder="Example: 14.6819",
    )

with gps_col2:

    longitude = st.text_input(
        "Longitude",
        placeholder="Example: 77.6006",
    )


# ============================================================
# DISPLAY SELECTED IMAGE
# ============================================================

if selected_image is not None:

    st.subheader(
        "🖼️ Selected Image"
    )

    image_col1, image_col2 = st.columns(
        [1, 1]
    )

    with image_col1:

        st.image(
            selected_image,
            caption="Input Crop Image",
            use_container_width=True,
        )

    with image_col2:

        width, height = selected_image.size

        st.write(
            f"**Image Width:** {width}px"
        )

        st.write(
            f"**Image Height:** {height}px"
        )

        quality_ok, quality_message = (
            check_image_quality(
                selected_image
            )
        )

        if quality_ok:

            st.success(
                "✅ " + quality_message
            )

        else:

            st.warning(
                "⚠️ " + quality_message
            )


# ============================================================
# PREDICT BUTTON
# ============================================================

if selected_image is not None:

    st.divider()

    predict_button = st.button(
        "🔍 Detect Crop Disease",
        type="primary",
        use_container_width=True,
    )

    if predict_button:

        quality_ok, quality_message = (
            check_image_quality(
                selected_image
            )
        )

        if not quality_ok:

            st.warning(
                quality_message
            )

        else:

            with st.spinner(
                "AI is analyzing the crop image..."
            ):

                try:

                    results = predict_disease(
                        selected_image
                    )

                    st.session_state.prediction_results = (
                        results
                    )

                    st.session_state.last_image = (
                        selected_image
                    )

                    st.session_state.last_mode = (
                        input_mode
                    )

                    save_prediction_history(
                        results,
                        input_mode,
                        latitude,
                        longitude,
                    )

                except Exception as error:

                    st.error(
                        "Prediction failed."
                    )

                    st.exception(
                        error
                    )


# ============================================================
# RESULTS
# ============================================================

results = st.session_state.prediction_results


if results:

    st.divider()

    st.header(
        "🧠 AI Detection Result"
    )

    primary = results[0]

    result_col1, result_col2, result_col3 = st.columns(
        3
    )

    with result_col1:

        st.metric(
            "🌱 Crop",
            primary["crop"],
        )

    with result_col2:

        st.metric(
            "🦠 Disease",
            primary["disease"],
        )

    with result_col3:

        st.metric(
            "🎯 Confidence",
            f"{primary['confidence']:.2f}%",
        )

    if primary["disease"].lower() == "healthy":

        st.success(
            "✅ The AI predicts that the crop leaf is healthy."
        )

    else:

        st.warning(
            "⚠️ A possible crop disease or stress condition "
            "has been detected."
        )


    # ========================================================
    # SOLUTION
    # ========================================================

    st.subheader(
        "💡 Recommended Solution"
    )

    st.write(
        primary["solution"]
    )

    st.subheader(
        "🛡️ Prevention"
    )

    st.write(
        primary["prevention"]
    )


    # ========================================================
    # TOP 5
    # ========================================================

    st.subheader(
        "🏆 Top 5 Predictions"
    )

    top5_dataframe = pd.DataFrame(
        [
            {
                "Rank": index + 1,
                "Crop": result["crop"],
                "Disease": result["disease"],
                "Confidence": (
                    f"{result['confidence']:.2f}%"
                ),
                "Status": result["status"],
            }

            for index, result in enumerate(
                results
            )
        ]
    )

    st.dataframe(
        top5_dataframe,
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # CONFIDENCE BARS
    # ========================================================

    st.subheader(
        "📊 Confidence"
    )

    confidence_dataframe = pd.DataFrame(
        {
            "Prediction": [
                (
                    f"{result['crop']} - "
                    f"{result['disease']}"
                )

                for result in results
            ],

            "Confidence": [
                result["confidence"]

                for result in results
            ],
        }
    )

    st.bar_chart(
        confidence_dataframe.set_index(
            "Prediction"
        )
    )


    # ========================================================
    # GPS RESULT
    # ========================================================

    st.subheader(
        "📍 GPS Information"
    )

    gps_result_col1, gps_result_col2 = (
        st.columns(2)
    )

    with gps_result_col1:

        st.write(
            f"**Latitude:** "
            f"{latitude if latitude else 'Not provided'}"
        )

    with gps_result_col2:

        st.write(
            f"**Longitude:** "
            f"{longitude if longitude else 'Not provided'}"
        )


    # ========================================================
    # PDF REPORT
    # ========================================================

    st.subheader(
        "📄 PDF Report"
    )

    try:

        pdf_bytes = create_pdf_report(
            st.session_state.last_image,
            results,
            st.session_state.last_mode,
            latitude,
            longitude,
        )

        report_filename = (
            "crop_disease_report_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            + ".pdf"
        )

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=report_filename,
            mime="application/pdf",
            use_container_width=True,
        )

    except Exception as error:

        st.error(
            "Unable to generate PDF report."
        )

        st.exception(error)


# ============================================================
# PREDICTION HISTORY
# ============================================================

st.divider()

st.header(
    "💾 Prediction History"
)

history_dataframe = load_history()


if history_dataframe.empty:

    st.info(
        "No prediction history available yet."
    )

else:

    st.dataframe(
        history_dataframe.tail(20),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# CONFIDENCE HISTORY
# ============================================================

st.subheader(
    "📈 Confidence History"
)


if not history_dataframe.empty:

    try:

        confidence_values = (
            history_dataframe[
                "Confidence"
            ]
            .astype(str)
            .str.replace(
                "%",
                "",
                regex=False,
            )
            .astype(float)
        )

        confidence_chart = pd.DataFrame(
            {
                "Confidence (%)": confidence_values
            }
        )

        st.line_chart(
            confidence_chart
        )

    except Exception:

        st.info(
            "Confidence history is not available."
        )


# ============================================================
# DRONE SYSTEM ARCHITECTURE
# ============================================================

st.divider()

st.header(
    "🚁 Drone System Architecture"
)

st.markdown(
    """
    **🚁 Quadcopter**   
    ↓   
    **📷 Drone Camera**   
    ↓   
    **📡 RTSP / HTTP Camera Stream**   
    ↓   
    **💻 Laptop / Ground Station**   
    ↓   
    **🧠 AI Disease Detection**   
    ↓   
    **🌱 Crop + Disease + Confidence**   
    ↓   
    **📍 GPS Information**   
    ↓   
    **💾 Prediction History**
    """
)


# ============================================================
# SYSTEM INFORMATION
# ============================================================

with st.expander(
    "🔧 System Information"
):

    st.write(
        "**Model:** CropDiseaseMobileNetV2_81Classes"
    )

    st.write(
        f"**Number of classes:** {len(CLASS_NAMES)}"
    )

    st.write(
        "**Architecture:** MobileNetV2"
    )

    st.write(
        "**Input size:** 224 × 224 × 3"
    )

    st.write(
        "**Prediction output:** Top 5"
    )

    st.write(
        "**Preprocessing:** Built into saved model"
    )

    st.write(
        "**External preprocessing in app:** Disabled"
    )

    st.write(
        "**Training Stage 1:** Completed"
    )

    st.write(
        "**Stage 2 Fine-Tuning:** Disabled"
    )

    st.write(
        f"**Model path:** {MODEL_PATH}"
    )

    st.write(
        f"**Class names:** {CLASS_NAMES_PATH}"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Crop Disease Detection System | "
    "MobileNetV2 | 81 Classes | "
    "Prototype for agricultural decision support"
)

