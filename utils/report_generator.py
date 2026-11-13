import os
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

class MedicalReportPDFGenerator:
    """
    Generates a professional, clinical-grade multi-modal screening report
    in PDF format combining the original fundus photograph, Grad-CAM overlay,
    X-HGNN sub-hypergraph grounding, clinical concept radar, and VLT narrative.
    """
    @classmethod
    def generate_pdf_report(
        cls,
        output_pdf_path,
        case_id,
        prediction_label,
        confidence,
        original_img_path,
        gradcam_img_path,
        hgnn_img_path,
        radar_img_path,
        clinical_concepts_list,
        vlt_explanation_text,
        patient_ref="ANON-SCREENING"
    ):
        os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
        doc = SimpleDocTemplate(
            output_pdf_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0F172A"),
            fontName="Helvetica-Bold",
            spaceAfter=4
        )
        
        subtitle_style = ParagraphStyle(
            "DocSubTitle",
            parent=styles["Normal"],
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#64748B"),
            fontName="Helvetica",
            spaceAfter=10
        )
        
        section_style = ParagraphStyle(
            "SectionHeader",
            parent=styles["Heading2"],
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#1E293B"),
            fontName="Helvetica-Bold",
            spaceBefore=8,
            spaceAfter=6
        )
        
        body_style = ParagraphStyle(
            "ReportBody",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#334155"),
            fontName="Helvetica"
        )
        
        disclaimer_style = ParagraphStyle(
            "DisclaimerStyle",
            parent=styles["Normal"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#B91C1C"),
            fontName="Helvetica-Bold"
        )

        story = []

        # --- 1. Header Banner ---
        story.append(Paragraph("OPHTHALMIC RESEARCH AI SCREENING REPORT", title_style))
        story.append(Paragraph(
            "Explainable Glaucoma Detection System | EfficientNet-B3 + X-HGNN + VLT",
            subtitle_style
        ))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284C7"), spaceAfter=12))

        # --- 2. Metadata & Prediction Summary ---
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        is_glaucoma = (prediction_label == "Glaucoma")
        badge_bg = colors.HexColor("#FEE2E2") if is_glaucoma else colors.HexColor("#DCFCE7")
        badge_fg = "#991B1B" if is_glaucoma else "#166534"

        meta_data = [
            [
                Paragraph(f"<b>Case Identifier:</b> {case_id}", body_style),
                Paragraph(f"<b>Patient Reference:</b> {patient_ref}", body_style)
            ],
            [
                Paragraph(f"<b>Acquisition Timestamp:</b> {timestamp_str}", body_style),
                Paragraph(f"<b>AI Screening Result:</b> <font color='{badge_fg}'><b>{prediction_label.upper()}</b></font>", body_style)
            ],
            [
                Paragraph("<b>Imaging Modality:</b> Digital Color Retinal Fundus", body_style),
                Paragraph(f"<b>Model Confidence:</b> <b>{confidence*100:.1f}%</b>", body_style)
            ]
        ]
        
        meta_table = Table(meta_data, colWidths=[270, 270])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 10))

        # --- 3. Multi-Modal Visual Explanations ---
        story.append(Paragraph("1. MULTI-MODAL EXPLAINABILITY (VISUAL & STRUCTURAL)", section_style))
        
        # Prepare side-by-side images
        img_row1 = []
        if os.path.exists(original_img_path):
            img_row1.append(RLImage(original_img_path, width=170, height=170))
        else:
            img_row1.append(Paragraph("[Original Fundus Unavailable]", body_style))

        if os.path.exists(gradcam_img_path):
            img_row1.append(RLImage(gradcam_img_path, width=170, height=170))
        else:
            img_row1.append(Paragraph("[Grad-CAM Overlay Unavailable]", body_style))

        if os.path.exists(radar_img_path):
            img_row1.append(RLImage(radar_img_path, width=170, height=170))
        else:
            img_row1.append(Paragraph("[Concept Profile Unavailable]", body_style))

        img_table = Table([
            img_row1,
            [
                Paragraph("<b>(A) Preprocessed Fundus</b>", body_style),
                Paragraph("<b>(B) EfficientNet-B3 Grad-CAM</b>", body_style),
                Paragraph("<b>(C) Clinical Concept Radar</b>", body_style)
            ]
        ], colWidths=[180, 180, 180])
        img_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 1), (-1, 1), 6)
        ]))
        story.append(img_table)
        story.append(Spacer(1, 8))

        # Add X-HGNN Grounding Plot if available
        if os.path.exists(hgnn_img_path):
            story.append(Paragraph("<b>(D) X-HGNN Differentiable Sub-Hypergraph Grounding:</b>", body_style))
            story.append(Spacer(1, 4))
            hgnn_img = RLImage(hgnn_img_path, width=540, height=220)
            story.append(hgnn_img)
            story.append(Spacer(1, 10))

        # --- 4. Model-Derived Clinical Concepts ---
        story.append(Paragraph("2. MODEL-DERIVED CLINICAL RETINAL CONCEPTS", section_style))
        story.append(Paragraph(
            "<i>Note: Concept scores are model-derived visual indicators on a [0.0 - 1.0] scale, not physical micrometer measurements.</i>",
            body_style
        ))
        story.append(Spacer(1, 4))

        concept_table_data = [
            ["Concept Name", "Derived Score", "Screening Status", "Clinical Relevance Description"]
        ]
        for c in clinical_concepts_list:
            concept_table_data.append([
                c["title"],
                f"{c['score']:.3f}",
                c["status"],
                Paragraph(c["description"], body_style)
            ])

        ct = Table(concept_table_data, colWidths=[140, 60, 110, 230])
        ct.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FAFC")])
        ]))
        story.append(ct)
        story.append(Spacer(1, 10))

        # --- 5. Vision-Language Transformer (VLT) Clinical Narrative ---
        story.append(Paragraph("3. VISION-LANGUAGE TRANSFORMER (VLT) CLINICAL EXPLANATION", section_style))
        
        # Replace newlines with breaks
        narrative_paragraphs = vlt_explanation_text.split("\n\n")
        vlt_cells = []
        for p in narrative_paragraphs:
            if p.strip():
                clean_p = p.replace("\n", "<br/>")
                vlt_cells.append(Paragraph(clean_p, body_style))
                vlt_cells.append(Spacer(1, 3))

        vlt_container = Table([[vlt_cells]], colWidths=[540])
        vlt_container.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#94A3B8")),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(vlt_container)
        story.append(Spacer(1, 12))

        # --- 6. Medical Disclaimer ---
        disclaimer_box = Table([[
            Paragraph(
                "<b>RESEARCH SCREENING DISCLAIMER:</b> This report was generated by an automated "
                "experimental deep learning pipeline (Explainable Glaucoma Detection Using EfficientNet-B3, "
                "X-HGNN, and VLT). It is designed exclusively for clinical research and triage assistance, "
                "and DOES NOT constitute a definitive medical diagnosis. A comprehensive ophthalmological "
                "examination including IOP, OCT, and gonioscopy is required for formal diagnosis.",
                disclaimer_style
            )
        ]], colWidths=[540])
        disclaimer_box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF2F2")),
            ("BOX", (0, 0), (-1, -1), 1.5, colors.HexColor("#DC2626")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(disclaimer_box)

        # Build document
        doc.build(story)
        return output_pdf_path
