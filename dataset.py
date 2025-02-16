import os
import pickle
import numpy as np
from ultralytics import YOLO
from lib.feature_extractor import FeatureExtractor
from typing import List

def group_images(base_dir):
    """Groups images by part number and timestamp"""
    groups = {}  # {part_num: [(timestamp, {view: path}), ...]}

    for part_num in os.listdir(base_dir):
        if part_num == '.DS_Store':
            continue
        groups[part_num] = []
        part_dir = os.path.join(base_dir, part_num)

        # Group by timestamp
        timestamps = {}  # {timestamp: {view: path}}
        for img in os.listdir(part_dir):
            if img == '.DS_Store':
                continue

            ts, view = img.split('_')  # e.g. "1739410505.655_front.jpg"
            view = view.split('.')[0]  # remove .jpg

            if ts not in timestamps:
                timestamps[ts] = {}
            timestamps[ts][view] = os.path.join(part_dir, img)

        groups[part_num].extend((ts, views) for ts, views in timestamps.items())

    return groups


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


def process_image_groups(extractor, groups):
    """
    Process a group of images (front, back, bottom)
    groups: # {part_num: [(timestamp, {view: path}), ...]}
    Returns embeddings, predictions, confidences, and paths for all views
    """
    examples = []

    for part_num in groups:
        print("Part", part_num)
        for (timestamp, image_paths_by_view) in groups[part_num]:
            print("  ", timestamp)
            examples.append(image_group_to_example(extractor, part_num, image_paths_by_view))

    return examples

def image_group_to_example(extractor, part_num, image_paths_by_view):
    views = ['front', 'back', 'bottom']
    embeddings = []
    predictions = []
    confidences = []
    paths = []
    for view in views:
        path = image_paths_by_view.get(view)
        feat, pred, conf = extractor.extract(path)

        embeddings.append(feat)
        predictions.append(pred)
        confidences.append(conf)
        paths.append(str(path))
    return Example(part_num, embeddings, predictions, confidences, paths)


dataset_name = "lego-multi-classify-01"

model = YOLO("lego-classify-05-447-fixed-num.pt")
extractor = FeatureExtractor(model)
groups = group_images('src/v1/')
examples = process_image_groups(extractor, groups)
print(examples)
with open(f"datasets/{dataset_name}.pkl", 'wb') as f:
    pickle.dump(examples, f)
