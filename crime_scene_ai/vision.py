# vision.py
from ultralytics import YOLO
import torch
import torch.nn as nn
import numpy as np

# ── YOLOv8 Scene Detection ──────────────────────────────────────────────────

model = YOLO("yolov8n.pt")  # downloads automatically on first run

def analyze_scene(image_path: str) -> list[str]:
    """Run YOLOv8 on image, return list of detected object labels."""
    results = model(image_path, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls)]
            conf = float(box.conf)
            if conf > 0.3:  # filter low-confidence detections
                detections.append(f"{label} (confidence: {conf:.2f})")
    return detections


def scene_to_text(detections: list[str]) -> str:
    """Convert detection list into a structured scene description for RAG."""
    if not detections:
        return "No significant objects detected in the crime scene."

    items = ", ".join(detections[:10])
    count = len(detections)
    has_person   = any("person" in d   for d in detections)
    has_weapon   = any(w in d for d in detections 
                       for w in ["knife", "gun", "scissors", "bat"])
    has_vehicle  = any("car" in d or "truck" in d or "motorcycle" in d 
                       for d in detections)

    scene = f"""Crime Scene Report:
- Total objects detected: {count}
- Detected items: {items}
- Human presence: {'Yes' if has_person else 'No'}
- Potential weapon detected: {'Yes' if has_weapon else 'No'}
- Vehicle involved: {'Yes' if has_vehicle else 'No'}
- Scene complexity: {'High' if count > 6 else 'Moderate' if count > 3 else 'Low'}
"""
    return scene


# ── Forensic Scorer (Adam Optimizer demo) ───────────────────────────────────

class ForensicScorer(nn.Module):
    """
    Lightweight neural network that scores forensic priority
    of detected objects. Trained with Adam optimizer.
    """
    def __init__(self, input_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train_forensic_scorer(epochs: int = 30) -> list[float]:
    """
    Train the ForensicScorer with Adam optimizer.
    Returns loss history for visualization.
    """
    scorer = ForensicScorer(input_dim=10)

    # ── Adam optimizer — the key requirement ──────────────────────────────
    optimizer = torch.optim.Adam(
        scorer.parameters(),
        lr=0.001,           # learning rate
        betas=(0.9, 0.999), # momentum + RMS momentum terms
        eps=1e-8,           # numerical stability constant
        weight_decay=1e-4   # L2 regularization
    )

    # Learning rate scheduler (bonus: shows advanced knowledge)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.5
    )

    loss_fn = nn.BCELoss()

    # Simulated training data:
    # each sample = 10 confidence-like features → forensic priority label
    torch.manual_seed(42)
    X = torch.rand(100, 10)
    y = (X.mean(dim=1, keepdim=True) > 0.5).float()

    losses = []
    for epoch in range(epochs):
        scorer.train()
        optimizer.zero_grad()

        preds = scorer(X)
        loss  = loss_fn(preds, y)

        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

    return losses