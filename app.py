import streamlit as st
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image

from predict import predict_disease
from disease_info import disease_info
from pdf_report import create_pdf_report
from image_quality import check_image_quality


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Crop Disease Detection",
    page_icon="🌿",
    layout="wide"
)


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

project_dir = Path(__file__).parent

history_file = (
    project_dir
    / "prediction_history.csv"
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title(
    "🌿 AI-Based Agricultural Crop Disease Detection System"
)

st.write(
    "Upload a crop leaf image and use AI to identify "
    "possible crop diseases."
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("🌱 About")

    st.write(
        "This system uses an AI image classification "
        "model to identify diseases from crop leaf images."
    )

    st.subheader("Supported Crops")

    st.write("🍅 Tomato")
    st.write("🥔 Potato")
    st.write("🫑 Bell Pepper")

    st.subheader("AI Model")

    st.write(
        "MobileNetV2 Transfer Learning"
    )

    st.subheader("Validation Accuracy")

    st.write(
        "91.93%"
    )

    st.info(
        "AI predictions are intended as decision-support "
        "information. Confirm important diagnoses with a "
        "qualified agricultural expert."
    )


# ---------------------------------------------------------
# UPLOAD SECTION
# ---------------------------------------------------------

st.header("📷 Upload Crop Leaf")

upload_col, info_col = st.columns(
    [2, 1]
)


with upload_col:

    uploaded_file = st.file_uploader(
        "Choose a crop leaf image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


with info_col:

    st.info(
        "For better results, upload a clear, "
        "well-lit close-up image of a crop leaf."
    )


# ---------------------------------------------------------
# IMAGE DISPLAY
# ---------------------------------------------------------

image = None

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    st.image(
        image,
        caption="Uploaded Crop Leaf",
        use_container_width=True
    )


# ---------------------------------------------------------
# ANALYZE BUTTON
# ---------------------------------------------------------

analyze_button = st.button(
    "🔍 Analyze Crop Leaf",
    type="primary"
)


# ---------------------------------------------------------
# AI ANALYSIS
# ---------------------------------------------------------

if uploaded_file is not None and analyze_button:

    # -----------------------------------------------------
    # IMAGE QUALITY CHECK
    # -----------------------------------------------------

    image_quality_ok, quality_warnings = (
        check_image_quality(image)
    )

    if not image_quality_ok:

        st.error(
            "⚠️ Image Quality Problem"
        )

        for warning in quality_warnings:

            st.warning(
                warning
            )

        st.info(
            "Please upload a clear, well-lit "
            "crop leaf image and try again."
        )

        st.stop()


    # -----------------------------------------------------
    # AI PREDICTION
    # -----------------------------------------------------

    with st.spinner(
        "🤖 AI is analyzing the leaf..."
    ):

        predicted_class, confidence, top_predictions = (
            predict_disease(image)
        )


    # -----------------------------------------------------
    # LOW CONFIDENCE CHECK
    # -----------------------------------------------------

    confidence_threshold = 0.60

    if confidence < confidence_threshold:

        st.warning(
            "⚠️ AI confidence is low."
        )

        st.info(
            "The system could not identify the crop "
            "disease with sufficient confidence."
        )

        st.write(
            f"Current confidence: "
            f"{confidence * 100:.2f}%"
        )

        st.write(
            "Please upload a clear, close-up image "
            "of a supported crop leaf."
        )

        st.stop()


    # -----------------------------------------------------
    # GET DISEASE INFORMATION
    # -----------------------------------------------------

    if predicted_class not in disease_info:

        st.error(
            "Disease information is not available "
            "for this prediction."
        )

        st.stop()


    selected_disease = disease_info[
        predicted_class
    ]

    crop_name = selected_disease[
        "crop"
    ]

    disease_name = selected_disease[
        "disease"
    ]

    symptoms = selected_disease[
        "symptoms"
    ]

    management = selected_disease[
        "management"
    ]


    # -----------------------------------------------------
    # CURRENT DATE AND TIME
    # -----------------------------------------------------

    current_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    st.header(
        "🤖 AI Analysis Result"
    )

    result_col1, result_col2, result_col3 = (
        st.columns(3)
    )


    with result_col1:

        st.metric(
            "Crop",
            crop_name
        )


    with result_col2:

        st.metric(
            "Disease",
            disease_name
        )


    with result_col3:

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


    # -----------------------------------------------------
    # CONFIDENCE BAR
    # -----------------------------------------------------

    st.subheader(
        "🎯 Prediction Confidence"
    )

    st.progress(
        confidence
    )


    if confidence >= 0.80:

        st.success(
            "High confidence prediction"
        )

    elif confidence >= 0.60:

        st.warning(
            "Moderate confidence prediction"
        )

    else:

        st.error(
            "Low confidence prediction"
        )


    # -----------------------------------------------------
    # TOP 3 PREDICTIONS
    # -----------------------------------------------------

    st.subheader(
        "🔍 Top 3 AI Predictions"
    )

    for rank, (
        prediction_class,
        prediction_confidence
    ) in enumerate(
        top_predictions,
        start=1
    ):

        prediction_name = (
            prediction_class
            .replace(
                "___",
                " - "
            )
            .replace(
                "_",
                " "
            )
        )

        prediction_percentage = (
            prediction_confidence * 100
        )

        st.write(
            f"**{rank}. {prediction_name}** "
            f"— {prediction_percentage:.2f}%"
        )


    # -----------------------------------------------------
    # DISEASE INFORMATION
    # -----------------------------------------------------

    st.header(
        "🦠 Disease Information"
    )

    st.subheader(
        f"{crop_name} - {disease_name}"
    )


    # -----------------------------------------------------
    # SYMPTOMS
    # -----------------------------------------------------

    st.subheader(
        "🔎 Symptoms"
    )

    for symptom in symptoms:

        st.write(
            f"• {symptom}"
        )


    # -----------------------------------------------------
    # MANAGEMENT
    # -----------------------------------------------------

    st.subheader(
        "🌱 Management Recommendations"
    )

    for recommendation in management:

        st.write(
            f"• {recommendation}"
        )


    # -----------------------------------------------------
    # PDF REPORT
    # -----------------------------------------------------

    st.subheader(
        "📄 Disease Report"
    )

    confidence_text = (
        f"{confidence * 100:.2f}%"
    )

    pdf_data = create_pdf_report(
        image=image,
        crop_name=crop_name,
        disease_name=disease_name,
        confidence_text=confidence_text,
        symptoms=symptoms,
        management=management,
        current_time=current_time
    )


    st.download_button(
        label="📥 Download Disease Report",
        data=pdf_data,
        file_name=(
            "crop_disease_report.pdf"
        ),
        mime="application/pdf"
    )


    # -----------------------------------------------------
    # SAVE PREDICTION HISTORY
    # -----------------------------------------------------

    history_record = pd.DataFrame(
        [
            {
                "Date & Time": current_time,
                "Crop": crop_name,
                "Disease": disease_name,
                "Confidence": (
                    f"{confidence * 100:.2f}%"
                )
            }
        ]
    )


    if history_file.exists():

        try:

            old_history = pd.read_csv(
                history_file
            )

            history_dataframe = pd.concat(
                [
                    old_history,
                    history_record
                ],
                ignore_index=True
            )

        except Exception:

            history_dataframe = history_record

    else:

        history_dataframe = history_record


    history_dataframe.to_csv(
        history_file,
        index=False
    )


    # -----------------------------------------------------
    # DISCLAIMER
    # -----------------------------------------------------

    st.warning(
        "⚠️ Disclaimer: AI predictions are intended "
        "as decision-support information. Confirm the "
        "diagnosis with a qualified agricultural expert "
        "before making treatment decisions."
    )


# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------

st.header(
    "📊 Prediction Dashboard"
)


if history_file.exists():

    try:

        history_dataframe = pd.read_csv(
            history_file
        )

    except Exception:

        history_dataframe = pd.DataFrame()


else:

    history_dataframe = pd.DataFrame()


# ---------------------------------------------------------
# DASHBOARD METRICS
# ---------------------------------------------------------

if not history_dataframe.empty:

    total_predictions = len(
        history_dataframe
    )

    healthy_cases = (
        history_dataframe["Disease"]
        .astype(str)
        .str.lower()
        .eq("healthy")
        .sum()
    )

    diseased_cases = (
        total_predictions
        - healthy_cases
    )


    metric_col1, metric_col2, metric_col3 = (
        st.columns(3)
    )


    with metric_col1:

        st.metric(
            "Total Predictions",
            total_predictions
        )


    with metric_col2:

        st.metric(
            "Diseased Cases",
            diseased_cases
        )


    with metric_col3:

        st.metric(
            "Healthy Cases",
            healthy_cases
        )


    # -----------------------------------------------------
    # RECENT PREDICTIONS
    # -----------------------------------------------------

    st.subheader(
        "🕒 Recent Predictions"
    )

    st.dataframe(
        history_dataframe.tail(10),
        use_container_width=True
    )


    # -----------------------------------------------------
    # DISEASE DISTRIBUTION
    # -----------------------------------------------------

    st.subheader(
        "🦠 Disease Distribution"
    )

    disease_distribution = (
        history_dataframe[
            "Disease"
        ]
        .value_counts()
    )

    st.bar_chart(
        disease_distribution
    )


    # -----------------------------------------------------
    # CROP DISTRIBUTION
    # -----------------------------------------------------

    st.subheader(
        "🌾 Crop Distribution"
    )

    crop_distribution = (
        history_dataframe[
            "Crop"
        ]
        .value_counts()
    )

    st.bar_chart(
        crop_distribution
    )


    # -----------------------------------------------------
    # CLEAR HISTORY
    # -----------------------------------------------------

    st.subheader(
        "🗑️ Prediction History"
    )

    clear_history = st.button(
        "Clear Prediction History"
    )


    if clear_history:

        if history_file.exists():

            history_file.unlink()

        st.success(
            "Prediction history cleared."
        )

        st.rerun()


else:

    st.info(
        "No prediction history available yet. "
        "Upload a crop leaf and perform an analysis "
        "to see dashboard statistics."
    )