import torch
import torch.nn as nn
import torch.nn.functional as F

from .efficientnet_b3 import EfficientNetB3Backbone
from .hypergraph import RetinalNodeExtractor, HypergraphConstructor
from .x_hgnn import XHGNN
from .gumbel_selector import GumbelSubHypergraphSelector
from .clinical_extractor import ClinicalConceptExtractor
from .vlt import VisionLanguageTransformer

class ExplainableGlaucomaModel(nn.Module):
    """
    Unified Research Architecture:
    Explainable Glaucoma Detection Using EfficientNet-B3,
    Hypergraph Neural Networks, Gumbel-Softmax, and Vision-Language Transformers.
    """
    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        fine_tune=True,
        node_dim=256,
        hgnn_out_dim=64,
        total_hyperedges=40,
        top_k_edges=12,
        config_mode="full_xhgnn_vlt"
    ):
        super().__init__()
        self.num_classes = num_classes
        self.config_mode = config_mode  # Ablation mode
        
        # 1. EfficientNet-B3 Backbone
        self.backbone = EfficientNetB3Backbone(
            num_classes=num_classes,
            pretrained=pretrained,
            fine_tune=fine_tune
        )
        
        # 2. Localized Retinal Feature Node Extractor (8x8 grid -> N=64)
        self.node_extractor = RetinalNodeExtractor(
            in_channels=1536,
            node_dim=node_dim,
            grid_size=(8, 8)
        )
        
        # 3. Hypergraph Constructor (H in R^{B x N x E})
        self.hypergraph_constructor = HypergraphConstructor(
            num_nodes=64,
            knn_k=6,
            num_semantic=16
        )
        
        # 4. Explainable Hypergraph Neural Network (X-HGNN)
        self.x_hgnn = XHGNN(
            in_features=node_dim,
            hidden_dim=128,
            out_dim=hgnn_out_dim,
            dropout=0.2
        )
        
        # 5. Differentiable Sub-Hypergraph Selector (Gumbel-Softmax)
        self.gumbel_selector = GumbelSubHypergraphSelector(
            edge_dim=hgnn_out_dim,
            hidden_dim=64,
            top_k=top_k_edges
        )
        
        # 6. Clinically Interpretable Concept Extractor
        self.clinical_extractor = ClinicalConceptExtractor(
            node_dim=hgnn_out_dim,
            edge_dim=hgnn_out_dim,
            concept_emb_dim=64
        )
        
        # 7. Unified Glaucoma Classification Head
        # Fuses global CNN image features (1536) + hypergraph structural graph embedding (64)
        fusion_dim = 1536 + hgnn_out_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # 8. Vision-Language Transformer (VLT)
        self.vlt = VisionLanguageTransformer(
            visual_dim=hgnn_out_dim,
            concept_dim=64,
            d_model=128,
            nhead=4,
            num_layers=2
        )

    def forward(self, x, temperature=1.0, hard_selection=False):
        """
        Complete forward pass through the research architecture.
        Args:
            x: Input retinal fundus image tensor [B, 3, 300, 300]
            temperature: Gumbel-Softmax sampling temperature
            hard_selection: bool, discrete selection for eval/viz
        Returns:
            dict containing all predictions, features, explanations, and losses
        """
        B = x.size(0)
        
        # Step 1: EfficientNet-B3 feature extraction
        backbone_logits, feat_map, pooled_cnn = self.backbone(x)
        
        # If in backbone-only ablation mode:
        if self.config_mode == "efficientnet_b3":
            probs = F.softmax(backbone_logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1)
            return {
                "logits": backbone_logits,
                "probabilities": probs,
                "predicted_class": pred_idx,
                "confidence": probs[torch.arange(B), pred_idx],
                "feat_map": feat_map,
                "reg_loss": {"sparsity_loss": torch.tensor(0.0, device=x.device), "entropy_loss": torch.tensor(0.0, device=x.device)}
            }
        
        # Step 2: Extract localized retinal feature nodes
        nodes, coords = self.node_extractor(feat_map)  # [B, 64, 256], [64, 2]
        
        # Step 3: Construct full hypergraph incidence matrix
        H = self.hypergraph_constructor(nodes)         # [B, 64, 40]
        
        # Step 4: First X-HGNN forward pass to compute structural representations
        node_emb, edge_emb, graph_emb = self.x_hgnn(nodes, H)
        
        # Step 5: Differentiable Gumbel-Softmax Sub-Hypergraph Selection
        H_sub, edge_weights, topk_edges, reg_loss = self.gumbel_selector(
            H=H,
            edge_embeddings=edge_emb,
            temperature=temperature,
            hard=hard_selection
        )
        
        # Step 6: Second refined propagation through selected sub-hypergraph
        node_emb_sub, edge_emb_sub, graph_emb_sub = self.x_hgnn(nodes, H_sub)
        
        # Step 7: Clinically Interpretable Concepts
        concept_scores, concept_tokens = self.clinical_extractor(
            node_emb=node_emb_sub,
            H_sub=H_sub,
            edge_emb=edge_emb_sub
        )
        
        # Step 8: Unified Glaucoma Classification
        # Fuses CNN global pooled features + X-HGNN selected sub-hypergraph representation
        fused_features = torch.cat([pooled_cnn, graph_emb_sub], dim=-1)
        if fused_features.size(0) == 1 and self.training:
            self.eval()
            final_logits = self.classifier(fused_features)
            self.train()
        else:
            final_logits = self.classifier(fused_features)
        
        probs = F.softmax(final_logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1)
        conf = probs[torch.arange(B), pred_idx]
        
        # Step 9: Vision-Language Transformer contextual encoding
        vlt_context = self.vlt(
            visual_nodes=node_emb_sub,
            concept_tokens=concept_tokens,
            pred_class=pred_idx
        )
        
        return {
            "logits": final_logits,
            "probabilities": probs,
            "predicted_class": pred_idx,
            "confidence": conf,
            "backbone_logits": backbone_logits,
            "feat_map": feat_map,
            "nodes": nodes,
            "coords": coords,
            "H_full": H,
            "H_sub": H_sub,
            "edge_weights": edge_weights,
            "selected_topk": topk_edges,
            "node_emb": node_emb_sub,
            "edge_emb": edge_emb_sub,
            "graph_emb": graph_emb_sub,
            "concept_scores": concept_scores,
            "concept_tokens": concept_tokens,
            "vlt_context": vlt_context,
            "reg_loss": reg_loss
        }
