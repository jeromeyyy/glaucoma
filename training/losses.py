import torch
import torch.nn as nn
import torch.nn.functional as F

class GlaucomaMultiTaskLoss(nn.Module):
    """
    Joint multi-objective loss function:
    L_total = L_classification + lambda_sparse * L_sparsity + lambda_ent * L_entropy + lambda_concept * L_concept
    """
    def __init__(
        self,
        class_weights=None,
        sparsity_weight=0.05,
        entropy_weight=0.01,
        concept_weight=0.05
    ):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.entropy_weight = entropy_weight
        self.concept_weight = concept_weight
        self.cls_criterion = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, model_outputs, targets):
        """
        Args:
            model_outputs: dict from ExplainableGlaucomaModel
            targets: [B] ground-truth class labels (0=Normal, 1=Glaucoma)
        Returns:
            total_loss: scalar tensor for backpropagation
            loss_dict: dict of individual loss values for tracking
        """
        logits = model_outputs["logits"]
        cls_loss = self.cls_criterion(logits, targets)

        reg_loss = model_outputs.get("reg_loss", {})
        sparsity_loss = reg_loss.get("sparsity_loss", torch.tensor(0.0, device=logits.device))
        entropy_loss = reg_loss.get("entropy_loss", torch.tensor(0.0, device=logits.device))

        # Semi-supervised clinical concept consistency loss:
        # Normal fundus should correlate with low CDR proxy, Glaucoma with elevated CDR proxy
        concept_scores = model_outputs.get("concept_scores", {})
        if "cup_to_disc_ratio_proxy" in concept_scores:
            cdr = concept_scores["cup_to_disc_ratio_proxy"].squeeze(-1) # [B]
            target_cdr_indicator = targets.float() * 0.7 + (1.0 - targets.float()) * 0.35
            concept_loss = F.mse_loss(cdr, target_cdr_indicator)
        else:
            concept_loss = torch.tensor(0.0, device=logits.device)

        total_loss = (
            cls_loss +
            self.sparsity_weight * sparsity_loss +
            self.entropy_weight * entropy_loss +
            self.concept_weight * concept_loss
        )

        return total_loss, {
            "total_loss": total_loss.item(),
            "cls_loss": cls_loss.item(),
            "sparsity_loss": sparsity_loss.item() if hasattr(sparsity_loss, "item") else float(sparsity_loss),
            "entropy_loss": entropy_loss.item() if hasattr(entropy_loss, "item") else float(entropy_loss),
            "concept_loss": concept_loss.item() if hasattr(concept_loss, "item") else float(concept_loss)
        }
