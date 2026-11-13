import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class ClinicalConceptManager:
    """
    Manages presentation, status derivation, and radar visualization
    for the 5 model-derived clinical retinal concepts.
    """
    CONCEPT_SPECS = {
        "optic_disc_morphology": {
            "title": "Optic Disc Morphology",
            "normal_desc": "Regular circular/oval contour with well-defined margin.",
            "suspect_desc": "Irregular contour, peripapillary halo or disc tilting.",
            "threshold": 0.55
        },
        "optic_cup_characteristics": {
            "title": "Optic Cup Excavation",
            "normal_desc": "Shallow central depression with preserved physiological color.",
            "suspect_desc": "Deepened central excavation with noticeable pallor.",
            "threshold": 0.55
        },
        "cup_to_disc_ratio_proxy": {
            "title": "Cup-to-Disc Ratio Proxy (CDR-P)",
            "normal_desc": "Estimated CDR proxy <= 0.55 (within typical physiological range).",
            "suspect_desc": "Estimated CDR proxy > 0.55 (enlarged vertical/horizontal cup).",
            "threshold": 0.55
        },
        "neuroretinal_rim_integrity": {
            "title": "Neuroretinal Rim Thinning",
            "normal_desc": "Preserved rim thickness adhering to ISNT proportions.",
            "suspect_desc": "Increased thinning risk, potential focal notching at vertical poles.",
            "threshold": 0.55
        },
        "vessel_attenuation_changes": {
            "title": "Vascular Pattern Shift",
            "normal_desc": "Standard central emergence with symmetric arcade branching.",
            "suspect_desc": "Nasal displacement, bayoneting, or localized vessel narrowing.",
            "threshold": 0.55
        }
    }

    @classmethod
    def format_concepts_for_ui(cls, raw_scores):
        """
        Converts raw model concept outputs into UI-ready structured representations.
        Args:
            raw_scores: dict mapping concept_key -> float or scalar tensor
        Returns:
            list of dicts with title, score, status, description, badge_color
        """
        formatted = []
        for key, spec in cls.CONCEPT_SPECS.items():
            val = raw_scores.get(key, 0.5)
            if hasattr(val, "item"):
                val = val.item()
            val = float(val)
            
            is_suspect = val >= spec["threshold"]
            status = "Suspect / Elevated" if is_suspect else "Normal / Preserved"
            badge = "danger" if is_suspect else "success"
            desc = spec["suspect_desc"] if is_suspect else spec["normal_desc"]

            formatted.append({
                "key": key,
                "title": spec["title"],
                "score": round(val, 3),
                "score_pct": round(val * 100, 1),
                "status": status,
                "badge": badge,
                "description": desc,
                "type": "Model-Derived Concept"
            })
        return formatted

    @classmethod
    def generate_radar_chart(cls, raw_scores, save_path=None):
        """
        Generates a 5-axis clinical concept radar chart.
        """
        labels = [spec["title"].replace(" ", "\n") for spec in cls.CONCEPT_SPECS.values()]
        values = []
        for key in cls.CONCEPT_SPECS.keys():
            val = raw_scores.get(key, 0.5)
            if hasattr(val, "item"): val = val.item()
            values.append(float(val))

        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor("#0F172A")
        ax.set_facecolor("#1E293B")

        # Plot radar boundary and fill
        ax.plot(angles, values, color="#38BDF8", linewidth=2.5, linestyle="solid")
        ax.fill(angles, values, color="#38BDF8", alpha=0.35)

        # Baseline threshold circle at 0.55
        ref_circle = [0.55] * (num_vars + 1)
        ax.plot(angles, ref_circle, color="#EF4444", linewidth=1.5, linestyle="--", label="Suspect Threshold (0.55)")

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, color="#E2E8F0", fontsize=9, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], color="#94A3B8", fontsize=8)
        ax.grid(color="#334155", linestyle=":", alpha=0.8)
        ax.spines["polar"].set_color("#334155")

        plt.title("Model-Derived Clinical Concept Profile", color="#F8FAFC", fontsize=11, pad=20, fontweight="bold")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)

        return fig
