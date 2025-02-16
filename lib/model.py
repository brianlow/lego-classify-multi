import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiViewAttention(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=2, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # x shape: (batch_size, 3, embedding_dim)
        attended, _ = self.attention(x, x, x)
        attended = self.norm(attended)
        return attended + x  # residual connection

class MultiViewFusion(nn.Module):
    def __init__(self, embedding_dim=768, num_classes=60):
        super().__init__()
        self.attention = MultiViewAttention(embedding_dim)

        # Final classification layer for both class probabilities and confidence
        self.classifier = nn.Linear(embedding_dim * 3, num_classes + 1)

    def forward(self, embeddings):
        # embeddings shape: (batch_size, 3, embedding_dim)

        # Apply attention and get enhanced features
        enhanced = self.attention(embeddings)

        # Flatten the enhanced features
        flattened = enhanced.reshape(enhanced.size(0), -1)

        # Get logits and confidence
        outputs = self.classifier(flattened)
        class_logits = outputs[:, :-1]
        confidence = torch.sigmoid(outputs[:, -1])

        return class_logits, confidence

def fusion_loss(class_logits, confidence, targets):
    # Classification loss
    cls_loss = F.cross_entropy(class_logits, targets)

    # Confidence loss - use the max class probability as target
    probs = F.softmax(class_logits, dim=1)
    confidence_target = probs.max(dim=1).values.detach()  # detach to avoid affecting classification
    conf_loss = F.mse_loss(confidence, confidence_target)

    # Combine losses - weight confidence loss less since it's auxiliary
    total_loss = cls_loss + 0.1 * conf_loss
    return total_loss
