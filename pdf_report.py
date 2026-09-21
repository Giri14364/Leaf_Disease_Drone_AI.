from io import BytesIO

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as ReportImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle
)


def create_pdf_report(
    image,
    crop_name,
    disease_name,
    confidence_text,
    symptoms,
    management,
    current_time
):
    pdf_buffer = BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    # Styles
    title_style = ParagraphStyle(
        "TitleStyle",
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        spaceBefore=15,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "NormalStyle",
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        spaceAfter=7
    )

    disclaimer_style = ParagraphStyle(
        "DisclaimerStyle",
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        spaceBefore=10
    )

    story = []

    # Title
    story.append(
        Paragraph(
            "AI Crop Disease Detection Report",
            title_style
        )
    )

    # Image
    image_buffer = BytesIO()

    image.convert("RGB").save(
        image_buffer,
        format="JPEG"
    )

    image_buffer.seek(0)

    report_image = ReportImage(
        image_buffer,
        width=3.2 * inch,
        height=3.2 * inch
    )

    story.append(report_image)

    story.append(
        Spacer(1, 20)
    )

    # Prediction details
    prediction_data = [
        ["Date & Time", current_time],
        ["Crop", crop_name],
        ["Disease", disease_name],
        ["Confidence", confidence_text]
    ]

    prediction_table = Table(
        prediction_data,
        colWidths=[
            1.6 * inch,
            4.4 * inch
        ]
    )

    prediction_table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    "grey"
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    "#eeeeee"
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold"
                ),
                (
                    "FONTNAME",
                    (1, 0),
                    (1, -1),
                    "Helvetica"
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                )
            ]
        )
    )

    story.append(
        prediction_table
    )

    # Common symptoms
    story.append(
        Paragraph(
            "Common Symptoms",
            heading_style
        )
    )

    for symptom in symptoms:

        story.append(
            Paragraph(
                str(symptom),
                normal_style
            )
        )

    # Management recommendations
    story.append(
        Paragraph(
            "Management Recommendations",
            heading_style
        )
    )

    for recommendation in management:

        story.append(
            Paragraph(
                str(recommendation),
                normal_style
            )
        )

    # Disclaimer
    story.append(
        Paragraph(
            "<b>Disclaimer:</b> AI predictions are "
            "intended as decision-support information. "
            "Confirm the diagnosis with a qualified "
            "agricultural expert before making treatment "
            "decisions.",
            disclaimer_style
        )
    )

    # Build PDF
    document.build(
        story
    )

    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()