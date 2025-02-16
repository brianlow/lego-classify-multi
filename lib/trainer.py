import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from typing import List, Tuple

from lib.model import MultiViewFusion, fusion_loss

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def train_model(
    model: MultiViewFusion,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    patience: int = 7
):
    device = get_device()
    model = model.to(device)

    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=patience)

    # Training loop
    best_val_acc = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for embeddings, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            embeddings, labels = embeddings.to(device), labels.to(device)

            optimizer.zero_grad()
            class_logits, confidence = model(embeddings)
            loss = fusion_loss(class_logits, confidence, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate accuracy
            predictions = torch.argmax(class_logits, dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)

                class_logits, confidence = model(embeddings)
                loss = fusion_loss(class_logits, confidence, labels)

                val_loss += loss.item()

                predictions = torch.argmax(class_logits, dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        # Print progress
        print(f'Epoch {epoch+1}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model
