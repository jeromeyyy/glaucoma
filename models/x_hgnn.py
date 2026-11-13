import torch
import torch.nn as nn
import torch.nn.functional as F

class HypergraphConv(nn.Module):
    """
    Standard Hypergraph Convolutional Layer (HGNN):
    Z = sigma(D_v^{-1/2} * H * W_e * D_e^{-1} * H^T * D_v^{-1/2} * X * Theta)
    Operates on batched node representations X [B, N, d_in]
    and incidence matrix H [B, N, E].
    """
    def __init__(self, in_features, out_features, dropout=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, X, H):
        """
        Args:
            X: [B, N, in_features]
            H: [B, N, E] (incidence matrix, soft or hard)
        Returns:
            X_out: [B, N, out_features]
            E_out: [B, E, out_features] (hyperedge representations)
        """
        B, N, E = H.size()
        eps = 1e-7

        # Node degree D_v: sum over hyperedges [B, N]
        D_v = torch.sum(H, dim=2) + eps  # [B, N]
        D_v_inv_sqrt = torch.pow(D_v, -0.5)  # [B, N]

        # Hyperedge degree D_e: sum over nodes [B, E]
        D_e = torch.sum(H, dim=1) + eps  # [B, E]
        D_e_inv = torch.pow(D_e, -1.0)    # [B, E]

        # Step 1: D_v^{-1/2} * X
        X_norm = X * D_v_inv_sqrt.unsqueeze(-1)  # [B, N, in_features]

        # Step 2: H^T * (D_v^{-1/2} * X) -> node to hyperedge aggregation
        # H^T is [B, E, N], X_norm is [B, N, in_features] -> [B, E, in_features]
        node_to_edge = torch.bmm(H.transpose(1, 2), X_norm)

        # Step 3: D_e^{-1} * node_to_edge -> hyperedge feature representations
        edge_feat = node_to_edge * D_e_inv.unsqueeze(-1)  # [B, E, in_features]

        # Step 4: H * edge_feat -> hyperedge to node aggregation
        edge_to_node = torch.bmm(H, edge_feat)  # [B, N, in_features]

        # Step 5: D_v^{-1/2} * edge_to_node
        prop_feat = edge_to_node * D_v_inv_sqrt.unsqueeze(-1)  # [B, N, in_features]

        # Step 6: Linear projection Theta + bias
        # Flatten batch for BatchNorm1d: [B*N, out_features]
        projected = torch.matmul(prop_feat, self.weight) + self.bias  # [B, N, out_features]
        
        # Apply BatchNorm and LeakyReLU
        B, N, C = projected.shape
        proj_flat = projected.view(-1, C)
        proj_bn = self.bn(proj_flat).view(B, N, C)
        X_out = F.leaky_relu(proj_bn, 0.1)
        X_out = self.dropout(X_out)

        # Hyperedge embeddings projected to out_features
        E_projected = torch.matmul(edge_feat, self.weight)  # [B, E, out_features]

        return X_out, E_projected

class XHGNN(nn.Module):
    """
    Explainable Hypergraph Neural Network (X-HGNN).
    Stack of Hypergraph Convolution layers learning multi-scale
    retinal structure embeddings and hyperedge representations.
    """
    def __init__(self, in_features=256, hidden_dim=128, out_dim=64, dropout=0.2):
        super().__init__()
        self.conv1 = HypergraphConv(in_features, hidden_dim, dropout=dropout)
        self.conv2 = HypergraphConv(hidden_dim, out_dim, dropout=dropout)
        
        # Global readout: aggregates node embeddings into graph-level representation
        self.readout_fc = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, nodes, H):
        """
        Args:
            nodes: [B, N, in_features]
            H: [B, N, E]
        Returns:
            node_emb: [B, N, out_dim]
            edge_emb: [B, E, out_dim]
            graph_emb: [B, out_dim]
        """
        x1, e1 = self.conv1(nodes, H)
        x2, e2 = self.conv2(x1, H)

        # Global mean pooling over nodes
        graph_emb = self.readout_fc(torch.mean(x2, dim=1))  # [B, out_dim]

        return x2, e2, graph_emb
