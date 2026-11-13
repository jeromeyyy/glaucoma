import os
import sys

# Ensure project root is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Silence third-party network checks and set cache
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["MPLCONFIGDIR"] = os.path.join(ROOT_DIR, ".matplotlib_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

modules = [
    "config",
    "utils.logging",
    "utils.seed",
    "utils.report_generator",
    "preprocessing.transforms",
    "preprocessing.augmentation",
    "preprocessing.preprocessing",
    "models.efficientnet_b3",
    "models.hypergraph",
    "models.x_hgnn",
    "models.gumbel_selector",
    "models.clinical_extractor",
    "models.vlt",
    "models.baseline_cnn",
    "models.full_model",
    "datasets.sample_generator",
    "datasets.dataset_loader",
    "training.losses",
    "training.checkpoints",
    "training.validate",
    "training.train",
    "evaluation.metrics",
    "evaluation.plots",
    "evaluation.evaluate",
    "evaluation.comparison",
    "explainability.gradcam",
    "explainability.hypergraph_explainer",
    "explainability.clinical_concepts",
    "explainability.vlt_explainer"
]

failed = []
passed = []

for mod in modules:
    try:
        __import__(mod)
        passed.append(mod)
        print(f"  [PASS] {mod}")
    except Exception as e:
        failed.append((mod, str(e)))
        print(f"  [FAIL] {mod}: {e}")

print(f"\nAudit complete: {len(passed)} passed, {len(failed)} failed.")
if failed:
    sys.exit(1)
