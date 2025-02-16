import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict
from lib.example import Example


def prepare_data(examples: List[Example], batch_size: int) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Convert Examples to DataLoader and return part number mapping"""
    # Create mapping of part numbers to integer indices
    unique_parts = sorted(set(ex.part_num for ex in examples))
    part_to_idx = {part: idx for idx, part in enumerate(unique_parts)}

    # Stack embeddings and convert to tensors with explicit float32 dtype
    embeddings = torch.tensor(np.stack([ex.embeddings for ex in examples]), dtype=torch.float32)
    labels = torch.tensor([part_to_idx[ex.part_num] for ex in examples])

    # Split into train/val (80/20)
    indices = torch.randperm(len(examples))
    train_size = int(0.8 * len(examples))

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create datasets
    train_dataset = TensorDataset(
        embeddings[train_indices],
        labels[train_indices]
    )
    val_dataset = TensorDataset(
        embeddings[val_indices],
        labels[val_indices]
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, part_to_idx
