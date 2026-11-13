import torch
import torch.nn as nn
import torch.nn.functional as F

class ClinicalConceptExtractor(nn.Module):
    """
    Extracts model-derived clinically interpretable concepts from
    the selected sub-hypergraph feature representations.
    
    IMPORTANT RESEARCH NOTE:
    These are model-derived visual concept proxies estimated from deep features
    and sub-hypergraph structure, NOT direct anatomical physical measurements.
    """
    CONCEPT_NAMES = [
        "optic_disc_morphology",       # Overall shape and margin regularity
        "optic_cup_characteristics",    # Central excavation and cup enlargement
        "cup_to_disc_ratio_proxy",      # Model-derived CDR proxy
        "neuroretinal_rim_integrity",   # ISNT compliance and rim thinning
        "vessel_attenuation_changes"    # Retinal vessel narrowing and nasal displacement
    ]

    CONCEPT_LABELS = {
        "optic_disc_morphology": "Optic Disc Morphology (Model-Derived)",
        "optic_cup_characteristics": "Optic Cup Excavation (Model-Derived)",
        "cup_to_disc_ratio_proxy": "Cup-to-Disc Ratio Proxy (CDR-P) (Model-Derived)",
        "neuroretinal_rim_integrity": "Neuroretinal Rim Thinning Risk (Model-Derived)",
        "vessel_attenuation_changes": "Vascular Attenuation / Shifting (Model-Derived)"
    }

    def __init__(self, node_dim=64, edge_dim=64, concept_emb_dim=64):
        super().__init__()
        self.num_concepts = len(self.CONCEPT_NAMES)
        self.concept_emb_dim = concept_emb_dim

        # Concept-specific projection heads from pooled sub-hypergraph features
        in_dim = node_dim + edge_dim
        
        # Probes for each of the 5 concepts
        self.concept_probes = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(in_dim, concept_emb_dim),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(concept_emb_dim, 1),
                nn.Sigmoid()
            ) for name in self.CONCEPT_NAMES
        })

        # Concept embedding projection (for conditioning VLT)
        self.concept_embedders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(1 + concept_emb_dim, concept_emb_dim),
                nn.LayerNorm(concept_emb_dim)
            ) for name in self.CONCEPT_NAMES
        })

        # Shared feature projector
        self.shared_proj = nn.Sequential(
            nn.Linear(in_dim, concept_emb_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, node_emb, H_sub, edge_emb):
        """
        Args:
            node_emb: [B, N, node_dim] refined node features
            H_sub: [B, N, E] selected sub-hypergraph incidence
            edge_emb: [B, E, edge_dim] hyperedge features
        Returns:
            concept_scores: dict mapping concept name -> [B, 1] tensor in [0, 1]
            concept_tokens: [B, num_concepts, concept_emb_dim] tokens for VLT cross-attention
        """
        B = node_emb.size(0)
        
        # Sub-hypergraph weighted node pooling
        # Sum over selected hyperedges gives node participation in selected structure
        node_importance = torch.sum(H_sub, dim=-1, keepdim=True)  # [B, N, 1]
        node_weights = F.softmax(node_importance, dim=1)           # [B, N, 1]
        pooled_nodes = torch.sum(node_emb * node_weights, dim=1)   # [B, node_dim]

        # Sub-hypergraph edge pooling
        edge_weights = F.softmax(torch.sum(H_sub, dim=1, keepdim=False), dim=-1) # [B, E]
        pooled_edges = torch.bmm(edge_weights.unsqueeze(1), edge_emb).squeeze(1) # [B, edge_dim]

        # Fused sub-hypergraph representation
        fused = torch.cat([pooled_nodes, pooled_edges], dim=-1)  # [B, in_dim]
        shared_feat = self.shared_proj(fused)                     # [B, concept_emb_dim]

        concept_scores = {}
        token_list = []

        for name in self.CONCEPT_NAMES:
            score = self.concept_probes[name](fused)  # [B, 1] in [0, 1]
            concept_scores[name] = score
            
            # Form concept token for VLT: concatenate score and shared feature
            token_in = torch.cat([score, shared_feat], dim=-1) # [B, 1 + concept_emb_dim]
            token = self.concept_embedders[name](token_in)      # [B, concept_emb_dim]
            token_list.append(token.unsqueeze(1))               # [B, 1, concept_emb_dim]

        concept_tokens = torch.cat(token_list, dim=1)  # [B, 5, concept_emb_dim]

        return concept_scores, concept_tokens
