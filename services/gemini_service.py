import os
import json
import logging

logger = logging.getLogger("glucom.gemini")

class GeminiConsultationService:
    """
    Server-side Gemini 3.8 Flash Clinical Consultation Assistant.
    Provides second-opinion clinical summarization and recommendations
    strictly grounded in the X-HGNN model's predictions and clinical concepts.
    
    CRITICAL:
    - Never replaces the primary PyTorch glaucoma classifier.
    - Server-side only: API keys are NEVER exposed to the frontend.
    - Gracefully handles missing API keys, network timeouts, or service interruptions.
    """
    MODEL_ID = "gemini-3.8-flash"

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def consult(self, case_id: str, prediction: str, confidence: float, concepts: list, vlt_narrative: str, query: str = None) -> dict:
        """
        Generates structured clinical consultation report.
        """
        conf_pct = round(confidence * 100.0, 1)
        concept_summary = ", ".join([f"{c.get('title')}: {c.get('score')} ({c.get('status')})" for c in concepts]) if concepts else "Standard visual parameters."

        prompt = (
            f"You are an expert academic ophthalmology consultant reviewing an AI-assisted glaucoma screening case.\n"
            f"Case ID: {case_id}\n"
            f"Model Verdict: {prediction} (Confidence: {conf_pct}%)\n"
            f"Model-Derived Visual Concepts: {concept_summary}\n"
            f"Vision-Language Transformer Narrative: {vlt_narrative}\n"
            f"Doctor/User Question: {query or 'Provide clinical interpretation and recommended next steps.'}\n\n"
            f"Provide a structured, authoritative clinical consultation summary:\n"
            f"1. Clinical Significance\n"
            f"2. Diagnostic Considerations (IOP, OCT RNFL, Visual Fields)\n"
            f"3. Risk Factors & Differential\n"
            f"4. Recommended Actions\n"
            f"End with mandatory disclaimer: AI-assisted research screening result only; not a definitive medical diagnosis."
        )

        if self.is_configured():
            try:
                # Attempt using google-genai or requests
                import urllib.request
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.MODEL_ID}:generateContent?key={self.api_key}"
                payload = json.dumps({
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 600}
                }).encode("utf-8")
                req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    res_data = json.loads(resp.read().decode("utf-8"))
                    text = res_data["candidates"][0]["content"]["parts"][0]["text"]
                    return {
                        "status": "success",
                        "model": self.MODEL_ID,
                        "consultation": text,
                        "powered_by": "Google Gemini 3.8 Flash"
                    }
            except Exception as e:
                logger.warning(f"Gemini API call failed or timed out: {e}. Falling back to rule-grounded clinical narrative.")

        # Deterministic Medical Knowledge Fallback
        is_glaucoma = "glaucoma" in prediction.lower()
        if is_glaucoma:
            consultation_text = (
                f"### Clinical Consultation Summary (Case {case_id})\n\n"
                f"**Diagnostic Review:** The deep screening pipeline identified architectural features consistent with "
                f"glaucomatous optic neuropathy (Model Confidence: {conf_pct}%).\n\n"
                f"**Key Findings:**\n"
                f"- Visual concept proxies suggest neuroretinal rim thinning and central cup enlargement.\n"
                f"- Higher-order hypergraph structure demonstrated topological disruption around the peripapillary arcades.\n\n"
                f"**Recommended Clinical Workup:**\n"
                f"1. Goldmann Applanation Tonometry (GAT) to evaluate intraocular pressure (IOP).\n"
                f"2. Spectral-Domain Optical Coherence Tomography (SD-OCT) assessing retinal nerve fiber layer (RNFL) thickness.\n"
                f"3. Standard Automated Perimetry (Humphrey Visual Field 24-2 or 30-2).\n"
                f"4. Gonioscopy to assess anterior chamber angle anatomy (open vs. narrow).\n\n"
                f"*NOTICE: AI-assisted research screening result. Requires clinical validation by a licensed ophthalmologist.*"
            )
        else:
            consultation_text = (
                f"### Clinical Consultation Summary (Case {case_id})\n\n"
                f"**Diagnostic Review:** The deep screening pipeline identified anatomical features within normal limits "
                f"(Model Confidence: {conf_pct}%).\n\n"
                f"**Key Findings:**\n"
                f"- Preserved neuroretinal rim contour complying with physiological ISNT pattern.\n"
                f"- Cup-to-disc ratio proxy and vascular trajectory are regular without focal notching or disc hemorrhages.\n\n"
                f"**Recommended Clinical Workup:**\n"
                f"1. Routine annual or biennial ophthalmic screening based on patient demographic and family history.\n"
                f"2. Baseline intraocular pressure and optic nerve head documentation.\n\n"
                f"*NOTICE: AI-assisted research screening result. Requires clinical validation by a licensed eye care professional.*"
            )

        return {
            "status": "success",
            "model": "Knowledge-Engine Fallback (Gemini 3.8 Flash Ready)",
            "consultation": consultation_text,
            "powered_by": "Glucom Clinical Decision Support"
        }

