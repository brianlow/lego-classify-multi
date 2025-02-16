import os


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
