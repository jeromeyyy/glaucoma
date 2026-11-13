import torch
import torch.nn as nn
import torch.nn.functional as F

class RetinalNodeExtractor(nn.Module):
    """
    Transforms intermediate 2D feature maps from EfficientNet-B3
    into localized retinal feature nodes with 2D spatial coordinates.
    Grid of 8x8 yields N=64 localized patch nodes.
    """
    def __init__(self, in_channels=1536, node_dim=256, grid_size=(8, 8)):
        super().__init__()
        self.grid_size = grid_size
        self.num_nodes = grid_size[0] * grid_size[1]
        self.node_dim = node_dim
        
        # Spatial adapt pooling to grid size
        self.adapt_pool = nn.AdaptiveAvgPool2d(grid_size)
        
        # Feature reduction: in_channels -> (node_dim - 2) so we can append 2D (x,y) coordinates
        feat_dim = node_dim - 2
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, feat_dim, kernel_size=1),
            nn.BatchNorm2d(feat_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Precompute normalized spatial coordinates [-1, 1] for each node
        ys = torch.linspace(-1.0, 1.0, grid_size[0])
        xs = torch.linspace(-1.0, 1.0, grid_size[1])
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # [N, 2]
        self.register_buffer("coords", coords)

    def forward(self, feat_map):
        """
        Args:
            feat_map: [B, C, H, W] from EfficientNet-B3
        Returns:
            nodes: [B, N, node_dim] localized node embeddings
            coords: [N, 2] spatial coordinates of nodes in [-1, 1]
        """
        B = feat_map.size(0)
        pooled = self.adapt_pool(feat_map)            # [B, C, 8, 8]
        projected = self.proj(pooled)                 # [B, feat_dim, 8, 8]
        nodes_feat = projected.flatten(2).permute(0, 2, 1)  # [B, N, feat_dim]
        
        # Expand spatial coordinates and concatenate
        coords_expanded = self.coords.unsqueeze(0).expand(B, -1, -1)  # [B, N, 2]
        nodes = torch.cat([nodes_feat, coords_expanded], dim=-1)       # [B, N, node_dim]
        return nodes, self.coords

class HypergraphConstructor(nn.Module):
    """
    Constructs a Hypergraph incidence matrix H in R^{B x N x E}
    where hyperedges connect MULTIPLE nodes beyond pairwise relationships.
    Contains:
    1. Spatial hyperedges (Quadrants, concentric rings, ISNT sectors) [E_spatial = 16]
    2. Semantic feature hyperedges (k-NN clusters in feature space) [E_semantic = 16]
    3. Vascular/Rim morphological hyperedges (arcade contours) [E_vascular = 8]
    Total E = 40 hyperedges.
    """
    def __init__(self, num_nodes=64, knn_k=6, num_semantic=16):
        super().__init__()
        self.num_nodes = num_nodes
        self.knn_k = knn_k
        self.num_semantic = num_semantic
        self.num_spatial = 16
        self.num_vascular = 8
        self.total_hyperedges = self.num_spatial + self.num_semantic + self.num_vascular
        
        # Build deterministic spatial and vascular hyperedge templates
        H_spatial = self._build_spatial_incidence()    # [N, 16]
        H_vascular = self._build_vascular_incidence()  # [N, 8]
        self.register_buffer("H_spatial", H_spatial)
        self.register_buffer("H_vascular", H_vascular)

    def _build_spatial_incidence(self):
        """
        Constructs 16 spatial hyperedges:
        - 4 quadrants (ST, SN, IT, IN)
        - 4 concentric rings (optic disc center, peripapillary, mid-periphery, outer)
        - 8 octant sectors (ISNT clinical sectors)
        """
        N = self.num_nodes
        G = int(N ** 0.5)  # 8
        H = torch.zeros(N, 16)
        
        # Grid indices
        for i in range(G):
            for j in range(G):
                idx = i * G + j
                # Normalized coordinates in [-1, 1]
                x = (j / (G - 1)) * 2.0 - 1.0
                y = (i / (G - 1)) * 2.0 - 1.0
                r = (x**2 + y**2)**0.5
                angle = torch.atan2(torch.tensor(y), torch.tensor(x)).item()
                
                # 4 Quadrants
                if x >= 0 and y >= 0: H[idx, 0] = 1.0
                elif x < 0 and y >= 0: H[idx, 1] = 1.0
                elif x < 0 and y < 0: H[idx, 2] = 1.0
                else: H[idx, 3] = 1.0
                
                # 4 Concentric Rings
                if r <= 0.35: H[idx, 4] = 1.0
                elif r <= 0.65: H[idx, 5] = 1.0
                elif r <= 0.95: H[idx, 6] = 1.0
                else: H[idx, 7] = 1.0
                
                # 8 Octants (ISNT sectors)
                sector = int(((angle + 3.14159) / (2 * 3.14159)) * 8) % 8
                H[idx, 8 + sector] = 1.0
                
        return H

    def _build_vascular_incidence(self):
        """
        Constructs 8 vascular/rim hyperedges tracing horizontal, vertical,
        and temporal arcades across the fundus.
        """
        N = self.num_nodes
        G = int(N ** 0.5)
        H = torch.zeros(N, 8)
        for i in range(G):
            for j in range(G):
                idx = i * G + j
                # Superior and Inferior vascular arcade bands
                if i in [1, 2]: H[idx, 0] = 1.0  # Superior arcade
                if i in [5, 6]: H[idx, 1] = 1.0  # Inferior arcade
                # Nasal and Temporal bands
                if j in [1, 2]: H[idx, 2] = 1.0  # Temporal rim
                if j in [5, 6]: H[idx, 3] = 1.0  # Nasal rim
                # Diagonal arcades
                if abs(i - j) <= 1: H[idx, 4] = 1.0
                if abs(i + j - (G - 1)) <= 1: H[idx, 5] = 1.0
                # Central cup-disc boundary
                if (2 <= i <= 5) and (2 <= j <= 5): H[idx, 6] = 1.0
                # Peripapillary crescent zone
                if (1 <= i <= 6) and (1 <= j <= 6) and not ((3 <= i <= 4) and (3 <= j <= 4)):
                    H[idx, 7] = 1.0
        return H

    def forward(self, nodes):
        """
        Args:
            nodes: [B, N, d] localized feature node vectors
        Returns:
            H: [B, N, E] complete hypergraph incidence matrix
        """
        B, N, d = nodes.size()
        device = nodes.device
        
        # 1. Expand static spatial and vascular hyperedges to batch
        H_sp = self.H_spatial.unsqueeze(0).expand(B, -1, -1).to(device)    # [B, N, 16]
        H_vasc = self.H_vascular.unsqueeze(0).expand(B, -1, -1).to(device) # [B, N, 8]
        
        # 2. Dynamic Semantic Hyperedges via k-NN on feature representations
        # Normalize node features for cosine similarity
        norm_nodes = F.normalize(nodes, p=2, dim=-1)  # [B, N, d]
        sim_matrix = torch.bmm(norm_nodes, norm_nodes.transpose(1, 2))  # [B, N, N]
        
        # Select 16 anchor nodes evenly spaced to create semantic clusters
        anchor_indices = torch.linspace(0, N - 1, self.num_semantic, dtype=torch.long, device=device)
        
        H_sem = torch.zeros(B, N, self.num_semantic, device=device)
        for e_idx, a_idx in enumerate(anchor_indices):
            # Top-k most similar nodes to this anchor form hyperedge e_idx
            sim_scores = sim_matrix[:, a_idx, :]  # [B, N]
            _, topk_indices = torch.topk(sim_scores, k=min(self.knn_k, N), dim=-1)  # [B, K]
            H_sem.scatter_(1, topk_indices.unsqueeze(-1), 1.0)
            
        # Combine all hyperedges: [B, N, E] (16 + 16 + 8 = 40)
        H = torch.cat([H_sp, H_sem, H_vasc], dim=-1)
        return H
