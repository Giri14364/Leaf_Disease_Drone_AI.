```python
from io import BytesIO

from reportlab.lib import colors
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
    TableStyle,
)


def create_pdf_report(
    image,
    crop_name,
    disease_name,
    confidence_text,
    symptoms,
    management,
    current_time,
):
    pdf_buffer = BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    # =========================================================
    # STYLES
    # =========================================================

    title_style = ParagraphStyle(
        "TitleStyle",
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#176B3A"),
        spaceAfter=5,
    )

    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#666666"),
        spaceAfter=18,
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#176B3A"),
        spaceBefore=15,
        spaceAfter=8,
    )

    normal_style = ParagraphStyle(
        "NormalStyle",
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#333333"),
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "BulletStyle",
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        leftIndent=12,
        firstLineIndent=-8,
        textColor=colors.HexColor("#333333"),
        spaceAfter=5,
    )

    story = []

    # =========================================================
    # TITLE
    # =========================================================

    story.append(
        Paragraph(
            "AI Crop Disease Detection Report",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "Agricultural Crop Health Analysis",
            subtitle_style,
        )
    )

    # =========================================================
    # UPLOADED / CAPTURED IMAGE
    # =========================================================

    image_buffer = BytesIO()

    image.convert("RGB").save(
        image_buffer,
        format="JPEG",
        quality=95,
    )

    image_buffer.seek(0)

    report_image = ReportImage(
        image_buffer,
        width=3.4 * inch,
        height=3.4 * inch,
    )

    image_table = Table(
        [[report_image]],
        colWidths=[6.0 * inch],
    )

    image_table.setStyle(
        TableStyle(
            [
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.8,
                    colors.HexColor("#D5D5D5"),
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#F8F8F8"),
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    10,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    10,
                ),
            ]
        )
    )

    story.append(image_table)

    story.append(
        Paragraph(
            "Uploaded / Captured Leaf Image",
            ParagraphStyle(
                "ImageCaption",
                parent=normal_style,
                alignment=TA_CENTER,
                textColor=colors.HexColor("#666666"),
                spaceBefore=6,
                spaceAfter=12,
            ),
        )
    )

    # =========================================================
    # DETECTION RESULT
    # =========================================================

    story.append(
        Paragraph(
            "Detection Result",
            heading_style,
        )
    )

    prediction_data = [
        ["Date & Time", str(current_time)],
        ["Crop", str(crop_name)],
        ["Detected Disease", str(disease_name)],
        ["Confidence", str(confidence_text)],
    ]

    prediction_table = Table(
        prediction_data,
        colWidths=[
            1.7 * inch,
            4.3 * inch,
        ],
    )

    prediction_table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#D0D0D0"),
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor("#EAF5EE"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (0, -1),
                    colors.HexColor("#176B3A"),
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "FONTNAME",
                    (1, 0),
                    (1, -1),
                    "Helvetica",
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
            ]
        )
    )

    story.append(prediction_table)

    # =========================================================
    # DISEASE EXPLANATION / SYMPTOMS
    # =========================================================

    story.append(
        Paragraph(
            "Disease Explanation & Symptoms",
            heading_style,
        )
    )

    if symptoms:
        for symptom in symptoms:
            story.append(
                Paragraph(
                    f"• {str(symptom)}",
                    bullet_style,
                )
            )
    else:
        story.append(
            Paragraph(
                "No disease explanation or symptom information available.",
                normal_style,
            )
        )

    # =========================================================
    # SOLUTION / MANAGEMENT
    # =========================================================

    story.append(
        Paragraph(
            "Recommended Solution & Management",
            heading_style,
        )
    )

    if management:
        for recommendation in management:
            story.append(
                Paragraph(
                    f"• {str(recommendation)}",
                    bullet_style,
                )
            )
    else:
        story.append(
            Paragraph(
                "No solution or management information available.",
                normal_style,
            )
        )

    # =========================================================
    # FOOTER
    # =========================================================

    story.append(
        Spacer(1, 18)
    )

    footer_style = ParagraphStyle(
        "FooterStyle",
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#777777"),
    )

    footer_table = Table(
        [
            [
                Paragraph(
                    "AI-Based Agricultural Crop Disease Detection System",
                    footer_style,
                )
            ]
        ],
        colWidths=[6.0 * inch],
    )

    footer_table.setStyle(
        TableStyle(
            [
                (
                    "LINEABOVE",
                    (0, 0),
                    (-1, 0),
                    0.5,
                    colors.HexColor("#D0D0D0"),
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
            ]
        )
    )

    story.append(footer_table)

    # =========================================================
    # BUILD PDF
    # =========================================================

    document.build(story)

    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()
```
