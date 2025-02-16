from typing import List
import numpy as np


class Example:
    """
    Represents a group of 3 images for a single part number
    """
    def __init__(self, part_num: str, embeddings: np.ndarray, predictions: List[float], confidences: List[float], paths: List[str]):
        self.part_num = part_num
        self.embeddings = embeddings
        self.predictions = predictions
        self.confidences = confidences
        self.paths = paths
