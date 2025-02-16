from pathlib import Path
import numpy as np
from PIL import Image

class FeatureExtractor:
    def __init__(self, model):
        self.model = model

    def extract(self, image_path):
        """
        Extract embedding, prediction, and confidence for a single image
        Returns:
            embedding: np.array # Extracted embeddings
            predicted_part: str # Predicted class
            confidence: float     # Prediction confidence
        """
        if image_path is None or not Path(image_path).exists():
            # Return zeros for missing/invalid images
            return np.zeros(768), -1, 0.0

        image = Image.open(image_path)

        # Get classification
        results = self.model.predict(image, embed=None, verbose=False)
        predicted_part = results[0].names[results[0].probs.top1]
        confidence = results[0].probs.top1conf.item()

        # Get embedding
        results = self.model.predict(image, embed=[9], verbose=False)
        embedding = results[0].cpu().numpy()

        return embedding, predicted_part, confidence

        # try:
        # except Exception as e:
        #     print(f"Error processing {image_path}: {str(e)}")
        #     return np.zeros(768), -1, 0.0
