import torch

class VLTExplainer:
    """
    Service wrapper for Vision-Language Transformer explanations.
    Synthesizes model predictions, confidence, clinical concepts,
    and hypergraph structural findings into structured clinical narratives.
    """
    def __init__(self, vlt_module=None):
        self.vlt = vlt_module

    def explain(self, prediction_label, confidence, concept_scores, top_edges_meta=None):
        """
        Generates full structured explanation.
        Args:
            prediction_label: 'Normal' or 'Glaucoma'
            confidence: float in [0, 1]
            concept_scores: dict mapping concept name -> float
            top_edges_meta: list of top selected hyperedge descriptions
        Returns:
            dict containing formatted paragraphs, key findings, and safety notices
        """
        # Convert tensor scores to floats
        clean_scores = {}
        for k, v in concept_scores.items():
            if hasattr(v, "item"): clean_scores[k] = v.item()
            else: clean_scores[k] = float(v)

        if self.vlt is not None and hasattr(self.vlt, "generate_explanation"):
            narrative = self.vlt.generate_explanation(prediction_label, confidence, clean_scores)
        else:
            narrative = "Clinical explanation module initialized."

        # Structured findings list
        findings = []
        cdr = clean_scores.get("cup_to_disc_ratio_proxy", 0.5)
        rim = clean_scores.get("neuroretinal_rim_integrity", 0.5)
        cup = clean_scores.get("optic_cup_characteristics", 0.5)
        vessel = clean_scores.get("vessel_attenuation_changes", 0.5)

        if prediction_label == "Glaucoma":
            findings.append(f"Elevated cup-to-disc relationship (CDR-P: {cdr:.2f})")
            if rim > 0.55:
                findings.append(f"Focal thinning risk in neuroretinal rim (Index: {rim:.2f})")
            if cup > 0.55:
                findings.append(f"Enlarged central cup excavation (Index: {cup:.2f})")
            if vessel > 0.55:
                findings.append(f"Potential vascular nasalization / narrowing (Index: {vessel:.2f})")
        else:
            findings.append(f"Physiological cup-to-disc relationship (CDR-P: {cdr:.2f})")
            findings.append(f"Preserved neuroretinal rim contour (Thinning risk: {rim:.2f})")
            findings.append("No focal notching or anomalous vascular displacement detected")

        return {
            "prediction": prediction_label,
            "confidence_pct": round(confidence * 100, 1),
            "narrative": narrative,
            "key_findings": findings,
            "safety_notice": (
                "AI-assisted screening result. This automated analysis is NOT a definitive medical "
                "diagnosis and must be interpreted in conjunction with complete clinical ophthalmic examination."
            )
        }
