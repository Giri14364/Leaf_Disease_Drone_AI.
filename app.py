
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
from reportlab.lib.utils import ImageReader


# ============================================================
# PAGE CONFIGURATION AND PATHS
# ============================================================
st.set_page_config(
    page_title="AI Crop Disease Detection",
    page_icon="🌱",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model" / "crop_disease_model_24classes.keras"
CLASS_NAMES_PATH = PROJECT_DIR / "model" / "class_names_24.txt"
HISTORY_DIR = PROJECT_DIR / "history"
HISTORY_FILE = HISTORY_DIR / "prediction_history.csv"

HISTORY_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
EXPECTED_CLASSES = 24
SUPPORTED_CROPS = {"guava", "mango", "neem"}

# Initial testing thresholds. Validate against real images before deployment.
MIN_CONFIDENCE = 55.0
MIN_TOP_TWO_GAP = 8.0

# ImageNet screening is supplementary and can make mistakes.
ENABLE_GENERAL_IMAGE_CHECK = True
GENERAL_IMAGE_REJECT_THRESHOLD = 0.50


# ============================================================
# DISEASE INFORMATION
# ============================================================
DISEASE_INFO = {
    "healthy": {
        "status": "Healthy",
        "solution": (
            "The image is classified as healthy. Continue regular irrigation, "
            "balanced fertilization, field monitoring and pest inspection."
        ),
        "prevention": (
            "Maintain suitable spacing, irrigation, nutrition and regular "
            "crop monitoring."
        ),
    },
}


# Keep this list conservative. ImageNet is not a dedicated leaf detector.
NON_PLANT_LABEL_TERMS = {
    "person", "bridegroom", "groom", "scuba diver", "military uniform",
    "dog", "cat", "bird", "hen", "rooster", "ostrich", "eagle", "owl",
    "frog", "turtle", "lizard", "snake", "crocodile", "monkey", "baboon",
    "chimpanzee", "gorilla", "elephant", "panda", "bear", "fox", "wolf",
    "lion", "tiger", "leopard", "horse", "zebra", "pig", "ox", "buffalo",
    "camel", "rabbit", "hare", "hamster", "squirrel", "mouse", "rat",
    "bat", "whale", "dolphin", "shark", "fish", "butterfly", "bee", "ant",
    "grasshopper", "cockroach", "spider", "scorpion", "aircraft", "airliner",
    "helicopter", "ship", "boat", "car", "sports car", "truck", "bus",
    "motorcycle", "bicycle", "tractor", "train", "computer", "laptop",
    "keyboard", "monitor", "printer", "telephone", "camera", "television",
    "chair", "sofa", "table", "desk", "bed", "book", "notebook", "bottle",
    "cup", "mug", "plate", "bowl", "spoon", "fork", "knife", "scissors",
    "hammer", "screwdriver", "umbrella", "shoe", "boot", "hat", "helmet",
    "shirt", "jean", "trousers", "dress", "coat", "jacket", "backpack",
    "handbag", "wallet", "watch", "clock", "sunglasses", "ball", "football",
    "basketball", "tennis ball", "guitar", "piano", "violin", "drum",
    "toy", "doll", "teddy", "building", "house", "church", "castle",
    "bridge", "street sign", "traffic light", "stop sign", "fire hydrant",
    "mailbox", "flag", "sculpture",
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
# LOAD MODEL AND CLASS NAMES
# ============================================================
@st.cache_data
def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_NAMES_PATH}"
        )

    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
        names = [line.strip() for line in file if line.strip()]

    if len(names) != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} classes, but found {len(names)}."
        )

    return names


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)

    if model.output_shape[-1] != EXPECTED_CLASSES:
        raise ValueError(
            f"The loaded model does not have {EXPECTED_CLASSES} output classes."
        )

    return model


@st.cache_resource
def load_general_image_model():
    """Load ImageNet MobileNetV2 for supplementary image screening."""
    if not ENABLE_GENERAL_IMAGE_CHECK:
        return None

    return tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=True,
        input_shape=(224, 224, 3),
    )


try:
    CLASS_NAMES = load_class_names()
    MODEL = load_model()
except Exception as error:
    st.error("Unable to load the crop disease model or class names.")
    st.exception(error)
    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def format_class_name(class_name):
    return class_name.replace("___", " - ").replace("_", " ").strip()


def extract_crop_and_disease(class_name):
    normalized = class_name.strip()

    if "___" in normalized:
        crop, disease = normalized.split("___", 1)
    elif "_" in normalized:
        crop, disease = normalized.split("_", 1)
    else:
        return normalized, "Unknown"

    return crop.replace("_", " ").strip(), disease.replace("_", " ").strip()


def get_disease_information(disease):
    if disease.strip().lower().startswith("healthy"):
        return DISEASE_INFO["healthy"]

    return {
        "status": "Possible disease or symptom detected",
        "solution": (
            "Inspect the plant and nearby leaves carefully. Remove badly "
            "affected material only when appropriate. Consult a local "
            "agricultural officer before applying pesticides or fungicides."
        ),
        "prevention": (
            "Maintain field sanitation, suitable plant spacing, balanced "
            "irrigation and nutrition, and inspect the crop regularly."
        ),
    }


def check_image_quality(image):
    rgb = image.convert("RGB")
    width, height = rgb.size

    if width < 100 or height < 100:
        return False, "Image resolution is too low. Please use a clearer image."

    stats = ImageStat.Stat(rgb.convert("L"))
    brightness = stats.mean[0]
    variance = stats.var[0]

    if brightness < 20:
        return False, "Image is too dark. Please provide a brighter image."

    if brightness > 245:
        return False, "Image is too bright. Please use a normally exposed image."

    if variance < 20:
        return False, "Image appears too blurry or lacks visual detail."

    return True, "Image quality is acceptable."


def not_categorized(
    confidence=None,
    reason="The image does not sufficiently match a supported category.",
):
    return [{
        "class_name": "not_categorized",
        "display_name": "Image is not categorized",
        "crop": "Image is not categorized",
        "disease": "Image is not categorized",
        "confidence": float(confidence or 0.0),
        "status": "Not categorized",
        "solution": (
            reason
            + " Please upload a clear Guava, Mango, or Neem leaf image."
        ),
        "prevention": (
            "Use a well-lit, close-up image with a leaf clearly visible "
            "against a simple background."
        ),
    }]


# ============================================================
# GENERAL IMAGE SCREENING
# ============================================================
def check_general_image_content(image):
    """
    Reject some obvious non-plant images.

    This is only a supplementary check. ImageNet is not trained to reliably
    distinguish supported crop leaves from every unrelated object.
    """
    if not ENABLE_GENERAL_IMAGE_CHECK:
        return True, ""

    try:
        general_model = load_general_image_model()

        if general_model is None:
            return True, ""

        rgb = image.convert("RGB").resize((224, 224))
        array = np.asarray(rgb, dtype=np.float32)

        # ImageNet MobileNetV2 expects pixels scaled to [-1, 1].
        # This preprocessing is ONLY for the separate ImageNet model.
        array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
        array = np.expand_dims(array, axis=0)

        predictions = general_model.predict(array, verbose=0)[0]
        best_index = int(np.argmax(predictions))
        best_score = float(predictions[best_index])

        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(
            np.expand_dims(predictions, axis=0),
            top=1,
        )[0]

        label = decoded[0][1].lower().replace("_", " ")

        is_non_plant = any(
            term in label for term in NON_PLANT_LABEL_TERMS
        )

        # Reject only when the top ImageNet result is a listed non-plant
        # category and its score exceeds the conservative threshold.
        if is_non_plant and best_score >= GENERAL_IMAGE_REJECT_THRESHOLD:
            return (
                False,
                f"The image appears to contain '{label}', "
                "rather than a supported crop leaf.",
            )

        return True, ""

    except Exception:
        # Fail open if ImageNet weights cannot be loaded or screening fails.
        # The main crop model can still attempt a prediction.
        return True, ""


# ============================================================
# CROP DISEASE PREDICTION
# ============================================================
def predict_disease(image):
    """Predict a supported crop class and reject uncertain results."""

    content_ok, content_message = check_general_image_content(image)

    if not content_ok:
        return not_categorized(reason=content_message)

    resized = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(resized, dtype=np.float32)
    array = np.expand_dims(array, axis=0)

    # IMPORTANT:
    # Do not call mobilenet_v2.preprocess_input() here.
    # The 24-class model already contains its own Rescaling layer:
    # scale=1/127.5 and offset=-1.0.
    probabilities = np.asarray(
        MODEL.predict(array, verbose=0)[0],
        dtype=np.float64,
    )

    if (
        probabilities.size != len(CLASS_NAMES)
        or not np.all(np.isfinite(probabilities))
    ):
        return not_categorized(
            reason="The crop model could not produce a valid prediction."
        )

    best_index = int(np.argmax(probabilities))
    best_probability = float(probabilities[best_index])
    confidence = best_probability * 100.0

    ordered = np.sort(probabilities)
    second_probability = float(ordered[-2]) if ordered.size > 1 else 0.0
    gap = (best_probability - second_probability) * 100.0

    class_name = CLASS_NAMES[best_index]
    crop, disease = extract_crop_and_disease(class_name)

    if crop.strip().lower() not in SUPPORTED_CROPS:
        return not_categorized(
            confidence,
            "The predicted category is outside the crops supported by this website.",
        )

    if confidence < MIN_CONFIDENCE:
        return not_categorized(
            confidence,
            "The prediction confidence is below the current testing threshold.",
        )

    if gap < MIN_TOP_TWO_GAP:
        return not_categorized(
            confidence,
            "The leading categories are too close to select confidently.",
        )

    information = get_disease_information(disease)

    return [{
        "class_name": class_name,
        "display_name": format_class_name(class_name),
        "crop": crop,
        "disease": disease,
        "confidence": confidence,
        "status": information["status"],
        "solution": information["solution"],
        "prevention": information["prevention"],
    }]


# ============================================================
# DRONE CAMERA CONNECTION
# ============================================================
def connect_to_drone(drone_url):
    if not drone_url or not drone_url.strip():
        return None, "Please enter the drone camera URL."

    camera = None

    try:
        camera = cv2.VideoCapture(drone_url.strip())
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not camera.isOpened():
            return (
                None,
                "Unable to connect to the drone camera. "
                "Check the URL and network connection.",
            )

        success, frame = camera.read()

        if not success or frame is None:
            return None, "Connected to the URL, but no video frame was received."

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return (
            Image.fromarray(frame_rgb).convert("RGB"),
            "Drone camera connected successfully.",
        )

    except Exception as error:
        return None, f"Drone connection failed: {error}"

    finally:
        if camera is not None:
            camera.release()


# ============================================================
# PREDICTION HISTORY
# ============================================================
def save_prediction_history(results, mode, latitude, longitude):
    if not results:
        return

    result = results[0]
    file_exists = HISTORY_FILE.exists()

    with HISTORY_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Date Time",
                "Mode",
                "Crop",
                "Disease",
                "Confidence",
                "Latitude",
                "Longitude",
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            mode,
            result["crop"],
            result["disease"],
            f"{result['confidence']:.2f}%",
            latitude,
            longitude,
        ])


def load_history():
    if not HISTORY_FILE.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(HISTORY_FILE)
    except Exception:
        return pd.DataFrame()


# ============================================================
# PDF REPORT
# ============================================================
def create_pdf_report(image, results, mode, latitude, longitude):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)

    page_width, page_height = A4
    left, right = 45, page_width - 45
    y = page_height - 48
    result = results[0]

    def draw_wrapped(
        text,
        x,
        y_pos,
        max_width,
        font_name="Helvetica",
        font_size=10,
        line_height=14,
    ):
        pdf.setFont(font_name, font_size)
        line = ""

        for word in str(text).split():
            candidate = f"{line} {word}".strip()

            if pdf.stringWidth(candidate, font_name, font_size) <= max_width:
                line = candidate
            else:
                if line:
                    pdf.drawString(x, y_pos, line)
                    y_pos -= line_height
                line = word

        if line:
            pdf.drawString(x, y_pos, line)
            y_pos -= line_height

        return y_pos

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(left, y, "AI Crop Disease Detection Report")
    y -= 28

    pdf.setFont("Helvetica", 9)
    pdf.drawString(
        left,
        y,
        "Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    y -= 15
    pdf.drawString(left, y, f"Detection mode: {mode}")
    y -= 28

    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(left, y, "Detection Result")
    y -= 22

    image_box_x = 335
    image_box_y = y - 135
    image_box_w = 210
    image_box_h = 150

    try:
        image_buffer = io.BytesIO()
        image.convert("RGB").save(
            image_buffer,
            format="JPEG",
            quality=90,
        )
        image_buffer.seek(0)

        pdf.drawImage(
            ImageReader(image_buffer),
            image_box_x,
            image_box_y,
            width=image_box_w,
            height=image_box_h,
            preserveAspectRatio=True,
            anchor="c",
            mask="auto",
        )

        pdf.setFont("Helvetica-Oblique", 8)
        pdf.drawCentredString(
            image_box_x + image_box_w / 2,
            image_box_y - 12,
            "Input image",
        )

    except Exception:
        pass

    y = draw_wrapped(
        f"Crop: {result['crop']}",
        left,
        y,
        260,
        font_size=11,
        line_height=19,
    )

    y = draw_wrapped(
        f"Disease: {result['disease']}",
        left,
        y,
        260,
        font_size=11,
        line_height=19,
    )

    if result.get("class_name") != "not_categorized":
        y = draw_wrapped(
            f"Confidence: {result['confidence']:.2f}%",
            left,
            y,
            260,
            font_size=11,
            line_height=19,
        )
    else:
        y = draw_wrapped(
            "Confidence: Not available (image not categorized)",
            left,
            y,
            260,
            font_size=10,
            line_height=19,
        )

    y = min(y - 10, image_box_y - 30)

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(left, y, "Recommended Solution")
    y -= 18

    y = draw_wrapped(
        result["solution"],
        left,
        y,
        right - left,
        font_size=9,
        line_height=13,
    ) - 8

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(left, y, "Prevention")
    y -= 18

    y = draw_wrapped(
        result["prevention"],
        left,
        y,
        right - left,
        font_size=9,
        line_height=13,
    )

    gps = []

    if latitude and str(latitude).strip():
        gps.append(f"Latitude: {str(latitude).strip()}")

    if longitude and str(longitude).strip():
        gps.append(f"Longitude: {str(longitude).strip()}")

    if gps:
        y -= 12
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(left, y, "GPS Information")
        y -= 15

        for item in gps:
            y = draw_wrapped(
                item,
                left,
                y,
                right - left,
                font_size=9,
                line_height=12,
            )

    pdf.save()
    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# HEADER AND INPUT
# ============================================================
st.title("🌱 AI Crop Disease Detection")

st.markdown("""
**AI-Based Agricultural Crop Disease Detection System**

Upload an image, use your laptop camera, or connect to a drone camera stream URL.
""")

st.caption(
    "Supported crops: Guava, Mango and Neem. "
    "Uncertain images may be marked as not categorized."
)

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


selected_image = None


# ============================================================
# UPLOAD IMAGE
# ============================================================
if input_mode == "📤 Upload Image":
    st.subheader("📤 Upload Crop Leaf Image")

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear leaf image.",
    )

    if uploaded_file is not None:
        try:
            selected_image = Image.open(uploaded_file).convert("RGB")
        except Exception as error:
            st.error("Unable to open the uploaded image.")
            st.exception(error)


# ============================================================
# LAPTOP CAMERA
# ============================================================
elif input_mode == "📷 Laptop Camera":
    st.subheader("📷 Laptop Camera")
    st.info("Allow camera access in your browser.")

    camera_file = st.camera_input("Take a picture of the crop leaf")

    if camera_file is not None:
        try:
            selected_image = Image.open(camera_file).convert("RGB")
        except Exception as error:
            st.error("Unable to read the camera image.")
            st.exception(error)


# ============================================================
# DRONE MODE
# ============================================================
else:
    st.subheader("🚁 Drone Camera")

    drone_input_type = st.radio(
        "Select drone input:",
        [
            "📁 Upload Drone Image",
            "🔗 Connect Drone Camera",
        ],
        horizontal=True,
    )

    if drone_input_type == "📁 Upload Drone Image":
        drone_file = st.file_uploader(
            "Upload drone image",
            type=["jpg", "jpeg", "png"],
            key="drone_upload",
        )

        if drone_file is not None:
            try:
                selected_image = Image.open(drone_file).convert("RGB")
            except Exception as error:
                st.error("Unable to open the drone image.")
                st.exception(error)

    else:
        st.info(
            "Enter the live camera stream URL provided by your drone or camera."
        )

        drone_url = st.text_input(
            "🔗 Drone Camera URL",
            placeholder="Example: rtsp://192.168.1.100:8554/live",
        )

        col_a, col_b = st.columns(2)

        with col_a:
            connect_button = st.button(
                "🔗 Connect to Drone",
                type="primary",
                use_container_width=True,
            )

        with col_b:
            capture_button = st.button(
                "📸 Capture Frame",
                use_container_width=True,
            )

        if connect_button or capture_button:
            if not drone_url.strip():
                st.warning("Please enter the drone camera URL.")
            else:
                with st.spinner("Connecting to drone camera..."):
                    drone_image, drone_message = connect_to_drone(drone_url)

                if drone_image is not None:
                    st.session_state.drone_frame = drone_image
                    st.session_state.drone_connected = True

                    st.success(
                        "📸 Frame captured successfully."
                        if capture_button
                        else drone_message
                    )
                else:
                    st.session_state.drone_connected = False
                    st.error(drone_message)

        if st.session_state.drone_connected:
            st.success("🟢 Drone camera connected")

        if st.session_state.drone_frame is not None:
            st.subheader("📡 Drone Camera Frame")

            st.image(
                st.session_state.drone_frame,
                caption="Latest Drone Camera Frame",
                use_container_width=True,
            )

            selected_image = st.session_state.drone_frame


# ============================================================
# GPS INFORMATION
# ============================================================
st.subheader("📍 GPS Information")

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
# IMAGE PREVIEW AND DETECTION
# ============================================================
if selected_image is not None:
    st.subheader("🖼️ Selected Image")

    image_col1, image_col2 = st.columns([1, 1])

    with image_col1:
        st.image(
            selected_image,
            caption="Input image",
            use_container_width=True,
        )

    with image_col2:
        width, height = selected_image.size

        st.write(f"**Image Width:** {width}px")
        st.write(f"**Image Height:** {height}px")

        quality_ok, quality_message = check_image_quality(selected_image)

        if quality_ok:
            st.success("✅ " + quality_message)
        else:
            st.warning("⚠️ " + quality_message)

    if st.button(
        "🔍 Detect Crop Disease",
        type="primary",
        use_container_width=True,
    ):
        quality_ok, quality_message = check_image_quality(selected_image)

        if not quality_ok:
            results = not_categorized(reason=quality_message)

            st.session_state.prediction_results = results
            st.session_state.last_image = selected_image.copy()
            st.session_state.last_mode = input_mode

            save_prediction_history(
                results,
                input_mode,
                latitude,
                longitude,
            )

        else:
            with st.spinner("Analyzing the crop leaf image..."):
                try:
                    results = predict_disease(selected_image)

                    st.session_state.prediction_results = results
                    st.session_state.last_image = selected_image.copy()
                    st.session_state.last_mode = input_mode

                    save_prediction_history(
                        results,
                        input_mode,
                        latitude,
                        longitude,
                    )

                except Exception as error:
                    st.error("Image analysis failed. Check the model and dependencies.")
                    st.exception(error)


# ============================================================
# DISPLAY RESULTS
# ============================================================
results = st.session_state.prediction_results

if results:
    st.divider()
    primary = results[0]

    if primary["class_name"] == "not_categorized":
        st.warning("🚫 Image is not categorized")
        st.write(primary["solution"])

    else:
        st.header("🧠 AI Detection Result")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("🌱 Crop", primary["crop"])

        with c2:
            st.metric("🦠 Disease", primary["disease"])

        with c3:
            st.metric(
                "🎯 Confidence",
                f"{primary['confidence']:.2f}%",
            )

        if primary["disease"].lower().startswith("healthy"):
            st.success("✅ The AI predicts that the crop is healthy.")
        else:
            st.warning(
                "⚠️ A possible crop disease or stress condition has been detected."
            )

        st.subheader("💡 Recommended Solution")
        st.write(primary["solution"])

        st.subheader("🛡️ Prevention")
        st.write(primary["prevention"])

    # --------------------------------------------------------
    # PDF REPORT
    # --------------------------------------------------------
    st.subheader("📄 PDF Report")

    try:
        if st.session_state.last_image is not None:
            pdf_bytes = create_pdf_report(
                st.session_state.last_image,
                results,
                st.session_state.last_mode,
                latitude,
                longitude,
            )

            filename = (
                "crop_disease_report_"
                + datetime.now().strftime("%Y%m%d_%H%M%S")
                + ".pdf"
            )

            st.download_button(
                "📥 Download PDF Report",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                use_container_width=True,
            )

    except Exception as error:
        st.error("Unable to generate PDF report.")
        st.exception(error)


# ============================================================
# PREDICTION HISTORY
# ============================================================
st.divider()
st.header("💾 Prediction History")

history_dataframe = load_history()

if history_dataframe.empty:
    st.info("No prediction history available yet.")
else:
    st.dataframe(
        history_dataframe.tail(20),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# CONFIDENCE HISTORY
# ============================================================
st.subheader("📈 Confidence History")

if not history_dataframe.empty and "Confidence" in history_dataframe.columns:
    try:
        values = (
            history_dataframe["Confidence"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .astype(float)
        )

        st.line_chart(
            pd.DataFrame({"Confidence (%)": values})
        )

    except Exception:
        st.info("Confidence history is not available.")


# ============================================================
# DRONE ARCHITECTURE
# ============================================================
st.divider()
st.header("🚁 Drone System Architecture")

st.markdown("""
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
""")


# ============================================================
# SYSTEM INFORMATION
# ============================================================
with st.expander("🔧 System Information"):
    st.write(f"**Number of crop classes:** {len(CLASS_NAMES)}")
    st.write("**Input size:** 224 × 224 × 3")
    st.write("**Prediction output:** Single best prediction with rejection checks")
    st.write("**Crop-model preprocessing:** Built into the saved model")
    st.write(f"**Minimum confidence:** {MIN_CONFIDENCE:.0f}%")
    st.write(
        f"**Minimum top-two gap:** {MIN_TOP_TWO_GAP:.0f} percentage points"
    )
    st.write(
        f"**General image screening:** "
        f"{'Enabled' if ENABLE_GENERAL_IMAGE_CHECK else 'Disabled'}"
    )
    st.write(f"**Model path:** {MODEL_PATH}")
    st.write(f"**Class names:** {CLASS_NAMES_PATH}")


st.divider()
st.caption("AI Crop Disease Detection System | 24 Classes")
