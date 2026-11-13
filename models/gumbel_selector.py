import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelSubHypergraphSelector(nn.Module):
    """
    Differentiable Sub-Hypergraph Selection Module using Gumbel-Softmax.
    Learns to select a compact, decision-relevant sub-hypergraph while
    preserving end-to-end differentiability during backpropagation.
    """
    def __init__(self, edge_dim=64, hidden_dim=64, top_k=12):
        super().__init__()
        self.edge_dim = edge_dim
        self.top_k = top_k
        
        # Scorer evaluates each hyperedge embedding to produce binary selection logits [not-selected, selected]
        self.scorer = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)  # [logit_reject, logit_select]
        )

    def forward(self, H, edge_embeddings, temperature=1.0, hard=False):
        """
        Args:
            H: [B, N, E] original incidence matrix
            edge_embeddings: [B, E, edge_dim] hyperedge representations from HGNN
            temperature: float > 0, Gumbel-Softmax temperature
            hard: bool, whether to use straight-through discrete one-hot during eval
        Returns:
            H_sub: [B, N, E] modulated sub-hypergraph incidence matrix
            selection_weights: [B, E] differentiable selection probability / mask
            selected_topk: [B, top_k] indices of top selected hyperedges for visualization
            reg_loss: dict containing sparsity_loss and entropy_loss
        """
        B, N, E = H.shape
        # Compute selection logits for all E hyperedges: [B, E, 2]
        logits = self.scorer(edge_embeddings)
        
        if self.training:
            # Differentiable Gumbel-Softmax sampling over the 2 states [reject, select]
            # gumbel_softmax returns [B, E, 2]
            sampled = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
            # Channel 1 corresponds to "selected" probability
            selection_weights = sampled[:, :, 1]  # [B, E] differentiable values in (0, 1)
        else:
            # During evaluation / inference: use softmax probabilities or hard top-k
            probs = F.softmax(logits / max(temperature, 0.1), dim=-1)[:, :, 1]
            if hard:
                # Top-K hard selection
                _, topk_idx = torch.topk(probs, k=min(self.top_k, E), dim=-1)
                mask = torch.zeros_like(probs)
                mask.scatter_(1, topk_idx, 1.0)
                selection_weights = mask
            else:
                selection_weights = probs

        # Modulate incidence matrix: H_sub_{b, v, e} = H_{b, v, e} * selection_weights_{b, e}
        # [B, 1, E] broadcast to [B, N, E]
        H_sub = H * selection_weights.unsqueeze(1)

        # Retrieve top-k hyperedges for qualitative visualization
        _, selected_topk = torch.topk(selection_weights, k=min(self.top_k, E), dim=-1)

        # Sparsity regularization: encourage selection of a compact subset (not all 40 hyperedges)
        sparsity_loss = torch.mean(selection_weights)
        
        # Entropy regularization: encourage decisive selection probabilities
        p_select = torch.clamp(selection_weights, 1e-6, 1.0 - 1e-6)
        entropy_loss = -torch.mean(p_select * torch.log(p_select) + (1 - p_select) * torch.log(1 - p_select))

        reg_loss = {
            "sparsity_loss": sparsity_loss,
            "entropy_loss": entropy_loss
        }

        return H_sub, selection_weights, selected_topk, reg_loss
