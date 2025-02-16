import os
import pickle
import numpy as np
from ultralytics import YOLO
from lib.example import Example
from lib.feature_extractor import FeatureExtractor
from typing import List

from lib.grouper import group_images

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

        print(len(feat))

        embeddings.append(feat)
        predictions.append(pred)
        confidences.append(conf)
        paths.append(str(path))
    return Example(part_num, embeddings, predictions, confidences, paths)


dataset_name = "lego-classify-multi-01"

model = YOLO("lego-classify-05-447-fixed-num.pt")
extractor = FeatureExtractor(model)
groups = group_images('src/v1/')
examples = process_image_groups(extractor, groups)
print(examples)
with open(f"datasets/{dataset_name}.pkl", 'wb') as f:
    pickle.dump(examples, f)
