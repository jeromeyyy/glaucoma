import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class EfficientNetB3Backbone(nn.Module):
    """
    EfficientNet-B3 Deep Feature Extractor & Classifier Backbone.
    Extracts deep 2D feature maps from intermediate conv stages
    for hypergraph node construction, as well as pooled embeddings.
    """
    def __init__(self, num_classes=2, pretrained=True, fine_tune=True):
        super().__init__()
        weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
        base = efficientnet_b3(weights=weights)
        
        self.features = base.features
        self.out_channels = 1536  # EfficientNet-B3 final conv channels
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.out_channels, num_classes)
        
        if not fine_tune:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward_features(self, x):
        """
        Extracts spatial feature map [B, 1536, H, W] from the backbone.
        For an input of 300x300, H=10, W=10.
        """
        return self.features(x)

    def forward(self, x):
        """
        Forward pass returning both classification logits and deep feature representations.
        Returns:
            logits: [B, num_classes]
            feature_map: [B, 1536, H, W]
            pooled_features: [B, 1536]
        """
        feat_map = self.features(x)
        pooled = self.avgpool(feat_map)
        flat = torch.flatten(pooled, 1)
        dropped = self.dropout(flat)
        logits = self.classifier(dropped)
        return logits, feat_map, flat
