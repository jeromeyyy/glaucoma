import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

class HypergraphExplainer:
    """
    Visualizes the selected sub-hypergraph and grounds higher-order
    retinal connections directly onto the fundus photograph.
    """
    HYPEREDGE_TYPES = {
        range(0, 4): ("Spatial Quadrant", "#3B82F6"),       # Blue
        range(4, 8): ("Concentric Zone", "#10B981"),        # Green
        range(8, 16): ("ISNT Sector", "#8B5CF6"),           # Purple
        range(16, 32): ("Semantic Cluster", "#F59E0B"),      # Amber
        range(32, 40): ("Vascular / Rim Arcade", "#EF4444") # Red
    }

    @classmethod
    def get_edge_metadata(cls, edge_idx):
        for r, (name, color) in cls.HYPEREDGE_TYPES.items():
            if edge_idx in r:
                return f"{name} #{edge_idx}", color
        return f"Hyperedge #{edge_idx}", "#6B7280"

    @classmethod
    def visualize_sub_hypergraph(
        cls,
        image_rgb,
        coords,
        H_sub,
        edge_weights,
        selected_topk,
        save_path=None
    ):
        """
        Creates a dual-panel figure:
        Left: Retinal fundus image with grounded sub-hypergraph nodes & hyperedges.
        Right: Importance ranking of selected hyperedges.
        
        Args:
            image_rgb: np.ndarray [H, W, 3] in [0, 255]
            coords: torch.Tensor or np.ndarray [N, 2] in [-1, 1]
            H_sub: torch.Tensor or np.ndarray [N, E]
            edge_weights: torch.Tensor or np.ndarray [E]
            selected_topk: list or 1D array of selected hyperedge indices
            save_path: str, destination image path
        Returns:
            fig: matplotlib figure
        """
        if hasattr(coords, "cpu"): coords = coords.cpu().numpy()
        if hasattr(H_sub, "cpu"): H_sub = H_sub.cpu().numpy()
        if hasattr(edge_weights, "cpu"): edge_weights = edge_weights.cpu().numpy()
        if hasattr(selected_topk, "cpu"): selected_topk = selected_topk.cpu().numpy()

        H_img, W_img = image_rgb.shape[:2]
        N, E = H_sub.shape

        # Convert normalized coordinates [-1, 1] to pixel coordinates [0, W], [0, H]
        # x is in [-1, 1] -> [0.1 * W, 0.9 * W]
        px = ((coords[:, 0] + 1.0) / 2.0) * (W_img * 0.8) + (W_img * 0.1)
        py = ((coords[:, 1] + 1.0) / 2.0) * (H_img * 0.8) + (H_img * 0.1)
        pixel_coords = np.stack([px, py], axis=-1)  # [N, 2]

        # Calculate node importance: weighted sum of incident selected hyperedges
        node_scores = np.zeros(N)
        for e in selected_topk:
            w = edge_weights[e]
            node_scores += H_sub[:, e] * w
        
        max_node_score = node_scores.max() if node_scores.max() > 0 else 1.0
        norm_node_scores = node_scores / max_node_score

        # Setup figure
        fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.2, 1.0]})
        fig.patch.set_facecolor("#0F172A")  # Dark medical background

        # --- Panel 1: Retinal Grounding ---
        ax_img.imshow(image_rgb)
        ax_img.set_title("X-HGNN Decision-Relevant Sub-Hypergraph", color="#F8FAFC", fontsize=12, pad=12, fontweight="bold")
        ax_img.axis("off")

        # Draw selected hyperedges as convex hulls or translucent overlays
        # Draw top 6 strongest hyperedges to avoid visual clutter
        for e_idx in selected_topk[:6]:
            w = edge_weights[e_idx]
            if w < 0.15: continue
            edge_name, color = cls.get_edge_metadata(e_idx)
            
            # Nodes belonging to this hyperedge
            member_nodes = np.where(H_sub[:, e_idx] > 0.1)[0]
            if len(member_nodes) >= 3:
                pts = pixel_coords[member_nodes]
                try:
                    hull = ConvexHull(pts)
                    hull_pts = pts[hull.vertices]
                    poly = Polygon(hull_pts, closed=True, facecolor=color, edgecolor=color, alpha=0.25 * float(w), linewidth=1.5)
                    ax_img.add_patch(poly)
                except Exception:
                    pass
            elif len(member_nodes) == 2:
                pts = pixel_coords[member_nodes]
                ax_img.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2.0 * float(w), alpha=0.6)

        # Plot all nodes with size & glow proportional to importance
        for i in range(N):
            score = norm_node_scores[i]
            size = 20 + 120 * score
            if score > 0.4:
                # Glowing halo
                ax_img.scatter(px[i], py[i], s=size * 2.2, color="#F59E0B", alpha=0.3, edgecolors="none")
                ax_img.scatter(px[i], py[i], s=size, color="#FBBF24", alpha=0.9, edgecolors="#FFFFFF", linewidth=1.0)
            else:
                ax_img.scatter(px[i], py[i], s=size * 0.7, color="#94A3B8", alpha=0.5, edgecolors="none")

        # --- Panel 2: Hyperedge Importance Bar Chart ---
        ax_bar.set_facecolor("#1E293B")
        top_k_indices = selected_topk[:10]
        bar_names = []
        bar_weights = []
        bar_colors = []

        for e in reversed(top_k_indices):
            name, col = cls.get_edge_metadata(e)
            bar_names.append(name)
            bar_weights.append(float(edge_weights[e]))
            bar_colors.append(col)

        y_pos = np.arange(len(bar_names))
        bars = ax_bar.barh(y_pos, bar_weights, color=bar_colors, height=0.65, edgecolor="#334155")
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(bar_names, color="#E2E8F0", fontsize=9)
        ax_bar.set_xlabel("Gumbel-Softmax Selection Weight", color="#94A3B8", fontsize=10)
        ax_bar.set_xlim(0, 1.05)
        ax_bar.set_title("Selected Hyperedges by Structural Saliency", color="#F8FAFC", fontsize=12, pad=12, fontweight="bold")
        ax_bar.tick_params(colors="#94A3B8")
        
        # Grid lines
        ax_bar.grid(axis="x", color="#334155", linestyle="--", alpha=0.7)

        # Add values on bars
        for bar in bars:
            w = bar.get_width()
            ax_bar.text(w + 0.02, bar.get_y() + bar.get_height()/2, f"{w:.2f}",
                        va="center", ha="left", color="#F8FAFC", fontsize=8, fontweight="bold")

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)

        return fig
