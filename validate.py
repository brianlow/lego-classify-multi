from typing import Dict, List, Tuple, Optional, NamedTuple
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import Counter, defaultdict
from PIL import Image, ImageDraw
from typing import List
import random

from lib.compensate import model_03_to_canonical, rebrickable_to_canonical
from lib.grouper import group_images

class PredictionResult(NamedTuple):
    true_class: str
    pred_class: str
    confidence: float
    views: Dict[str, str]
    view_used: Optional[str] = None  # Which view was used for the final prediction

class FusionEvaluator:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict_single_image(self, image_path: Optional[str]) -> Tuple[Optional[str], float]:
        """Predict class and confidence for a single image"""
        if image_path is None or not Path(image_path).exists():
            return None, 0.0

        # try:
        image = Image.open(image_path)
        results = self.model.predict(image, verbose=False)
        pred_class = results[0].names[results[0].probs.top1]
        pred_class = compensate(pred_class)
        confidence = results[0].probs.top1conf.item()
        return pred_class, confidence
        # except Exception as e:
        #     print(f"Error processing {image_path}: {str(e)}")
        #     return None, 0.0

    def evaluate_front_only(self, examples: List[Tuple[str, Dict[str, str]]]) -> Tuple[Dict, List[PredictionResult]]:
        """Baseline: Use only front view predictions"""
        correct = 0
        total = 0
        failed_predictions = 0
        all_results = []

        for true_class, views in examples:
            total += 1
            pred_class, confidence = self.predict_single_image(views.get('front'))

            if pred_class is None:
                failed_predictions += 1
                continue

            if pred_class == true_class:
                correct += 1

            result = PredictionResult(
                true_class=true_class,
                pred_class=pred_class,
                confidence=confidence,
                views=views,
                view_used='front'
            )
            all_results.append(result)

        return {
            'accuracy': correct / total if total > 0 else 0,
            'failed_predictions': failed_predictions,
            'total': len(examples)
        }, all_results

    def evaluate_first_available(self, examples: List[Tuple[str, Dict[str, str]]]) -> Tuple[Dict, List[PredictionResult]]:
        """Use first available view (front -> back -> bottom)"""
        correct = 0
        total = 0
        failed_predictions = 0
        view_used_counts = Counter()
        all_results = []

        for true_class, views in examples:
            # Try views in priority order
            for view in ['front', 'back', 'bottom']:
                if views.get(view):
                    pred_class, confidence = self.predict_single_image(views[view])
                    if pred_class is not None:
                        view_used_counts[view] += 1
                        total += 1
                        if pred_class == true_class:
                            correct += 1

                        result = PredictionResult(
                            true_class=true_class,
                            pred_class=pred_class,
                            confidence=confidence,
                            views=views,
                            view_used=view
                        )
                        all_results.append(result)
                        break
            else:
                failed_predictions += 1

        return {
            'accuracy': correct / total if total > 0 else 0,
            'failed_predictions': failed_predictions,
            'total': len(examples),
            'view_usage': dict(view_used_counts)
        }, all_results

    def get_all_predictions(self, views: Dict[str, str]) -> List[Tuple[str, str, float]]:
        """Get predictions for all available views"""
        predictions = []
        for view_name, path in views.items():
            pred_class, confidence = self.predict_single_image(path)
            if pred_class is not None:
                predictions.append((view_name, pred_class, confidence))
        return predictions

    def fuse_max_confidence(self, examples: List[Tuple[str, Dict[str, str]]]) -> Tuple[Dict, List[PredictionResult]]:
        """Use prediction with highest confidence across all views"""
        correct = 0
        total = 0
        failed_predictions = 0
        view_used_counts = Counter()
        all_results = []

        for true_class, views in examples:
            predictions = self.get_all_predictions(views)

            if not predictions:
                failed_predictions += 1
                continue

            # Take prediction with highest confidence
            view_name, pred_class, confidence = max(predictions, key=lambda x: x[2])
            view_used_counts[view_name] += 1

            total += 1
            if pred_class == true_class:
                correct += 1

            result = PredictionResult(
                true_class=true_class,
                pred_class=pred_class,
                confidence=confidence,
                views=views,
                view_used=view_name
            )
            all_results.append(result)

        return {
            'accuracy': correct / total if total > 0 else 0,
            'failed_predictions': failed_predictions,
            'total': len(examples),
            'view_usage': dict(view_used_counts)
        }, all_results

    def fuse_ensemble_voting(self, examples: List[Tuple[str, Dict[str, str]]]) -> Tuple[Dict, List[PredictionResult]]:
        """Use weighted voting with confidence thresholds"""
        confidence_threshold = 0.5  # Minimum confidence to consider a prediction

        correct = 0
        total = 0
        failed_predictions = 0
        view_used_counts = Counter()
        all_results = []

        for true_class, views in examples:
            predictions = self.get_all_predictions(views)

            if not predictions:
                failed_predictions += 1
                continue

            # Filter predictions by confidence threshold
            valid_predictions = [
                (view_name, pred_class, conf)
                for view_name, pred_class, conf in predictions
                if conf >= confidence_threshold
            ]

            if not valid_predictions:
                # Fall back to highest confidence if no predictions meet threshold
                view_name, pred_class, confidence = max(predictions, key=lambda x: x[2])
            else:
                # Weight votes by confidence
                class_votes = {}
                for view_name, pred_class, conf in valid_predictions:
                    class_votes[pred_class] = class_votes.get(pred_class, 0) + conf

                # Select class with highest weighted votes
                pred_class = max(class_votes.items(), key=lambda x: x[1])[0]

                # Find the view that contributed most to winning prediction
                view_entries = [
                    (v_name, conf) for v_name, p_class, conf in valid_predictions
                    if p_class == pred_class
                ]
                view_name, confidence = max(view_entries, key=lambda x: x[1])

            view_used_counts[view_name] += 1
            total += 1

            if pred_class == true_class:
                correct += 1

            result = PredictionResult(
                true_class=true_class,
                pred_class=pred_class,
                confidence=confidence,
                views=views,
                view_used=view_name
            )
            all_results.append(result)

        return {
            'accuracy': correct / total if total > 0 else 0,
            'failed_predictions': failed_predictions,
            'total': len(examples),
            'view_usage': dict(view_used_counts)
        }, all_results

    def fuse_sequential_confidence(self, examples: List[Tuple[str, Dict[str, str]]]) -> Tuple[Dict, List[PredictionResult]]:
        """Use sequential decision making with confidence thresholds"""
        high_confidence = 0.9  # Threshold for immediate acceptance
        low_confidence = 0.3   # Threshold for rejection

        correct = 0
        total = 0
        failed_predictions = 0
        view_used_counts = Counter()
        all_results = []

        for true_class, views in examples:
            # Try views in priority order
            best_prediction = None
            accumulated_evidence = {}

            for view in ['front', 'back', 'bottom']:
                if views.get(view):
                    pred_class, confidence = self.predict_single_image(views[view])

                    if pred_class is None:
                        continue

                    # Track prediction evidence
                    if pred_class not in accumulated_evidence:
                        accumulated_evidence[pred_class] = []
                    accumulated_evidence[pred_class].append((view, confidence))

                    # Update best prediction if needed
                    if (best_prediction is None or
                        confidence > best_prediction[2] or
                        (pred_class in accumulated_evidence and
                        len(accumulated_evidence[pred_class]) > 1)):
                        best_prediction = (view, pred_class, confidence)

                    # Accept high confidence predictions immediately
                    if confidence >= high_confidence:
                        break

                    # Reject low confidence predictions unless corroborated
                    if confidence <= low_confidence and len(accumulated_evidence.get(pred_class, [])) <= 1:
                        continue

            if best_prediction is None:
                failed_predictions += 1
                continue

            view_name, pred_class, confidence = best_prediction
            view_used_counts[view_name] += 1
            total += 1

            if pred_class == true_class:
                correct += 1

            result = PredictionResult(
                true_class=true_class,
                pred_class=pred_class,
                confidence=confidence,
                views=views,
                view_used=view_name
            )
            all_results.append(result)

        return {
            'accuracy': correct / total if total > 0 else 0,
            'failed_predictions': failed_predictions,
            'total': len(examples),
            'view_usage': dict(view_used_counts)
        }, all_results

    def predict_single_image_topk(self, image_path: Optional[str], k: int = 3) -> List[Tuple[str, float]]:
        """Predict top k classes and confidences for a single image"""
        if image_path is None or not Path(image_path).exists():
            return []

        image = Image.open(image_path)
        results = self.model.predict(image, verbose=False)

        # Get top k predictions
        probs = results[0].probs
        topk_indices = (-probs.data).argsort()[:k].tolist()
        predictions = [
            (compensate(results[0].names[idx]), probs.data[idx].item())
            for idx in topk_indices
        ]
        return predictions

    def fuse_topk_confidence(self, examples: List[Tuple[str, Dict[str, str]]], k: int = 3) -> Tuple[Dict, List[PredictionResult]]:
        """
        Fuse predictions by summing confidence scores of top k predictions across all views.
        For each view, get top k predictions and add their confidences to a running total.
        """
        correct = 0
        total = 0
        failed_predictions = 0
        view_used_counts = Counter()
        all_results = []

        for true_class, views in examples:
            # Accumulate confidence scores for each predicted class
            class_confidence = defaultdict(float)
            view_confidences = {}  # Track which view gave highest confidence for each class

            # Get top k predictions from each view
            for view_name, path in views.items():
                predictions = self.predict_single_image_topk(path, k=k)

                for pred_class, confidence in predictions:
                    # Add confidence to running total for this class
                    class_confidence[pred_class] += confidence

                    # Track which view gave highest confidence for this prediction
                    if pred_class not in view_confidences or confidence > view_confidences[pred_class][1]:
                        view_confidences[pred_class] = (view_name, confidence)

            if not class_confidence:
                failed_predictions += 1
                continue

            # Select class with highest total confidence
            pred_class = max(class_confidence.items(), key=lambda x: x[1])[0]
            total_confidence = class_confidence[pred_class]
            view_name = view_confidences[pred_class][0]

            view_used_counts[view_name] += 1
            total += 1

            if pred_class == true_class:
                correct += 1

            result = PredictionResult(
                true_class=true_class,
                pred_class=pred_class,
                confidence=total_confidence,
                views=views,
                view_used=view_name
            )
            all_results.append(result)

        return {
            'accuracy': correct / total if total > 0 else 0,
            'failed_predictions': failed_predictions,
            'total': len(examples),
            'view_usage': dict(view_used_counts)
        }, all_results

def create_error_visualization(results: List[PredictionResult], strategy_name: str, num_examples: int = 16):
    """Create a grid of misclassified examples with instruction manual images for both true and predicted classes"""
    # Filter for incorrect predictions
    errors = [r for r in results if r.true_class != r.pred_class]

    # Randomly sample errors
    sample_size = min(num_examples, len(errors))
    samples = random.sample(errors, sample_size)

    # Calculate grid size
    cols = 4
    rows = (sample_size + cols - 1) // cols

    # Create canvas
    # Each example gets 5 images side by side (true instruction, 3 views, pred instruction), plus padding and text
    img_size = 100
    example_width = 5 * img_size + 60  # 5 images plus padding
    example_height = img_size + 40  # image plus text
    width = cols * example_width + 20
    height = rows * example_height + 20

    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)

    for idx, result in enumerate(samples):
        row = idx // cols
        col = idx % cols
        x = col * example_width + 10
        y = row * example_height + 10

        # Add true class instruction manual image first
        instruction_path = f"../lego-part-library/instruction/{result.true_class}.png"
        try:
            instruction_img = Image.open(instruction_path)
            instruction_img = instruction_img.resize((img_size, img_size))
            canvas.paste(instruction_img, (x, y))
        except Exception as e:
            print(f"Error loading true class instruction image for {result.true_class}: {str(e)}")
            # Create a placeholder for missing instruction image
            placeholder = Image.new('RGB', (img_size, img_size), 'lightgray')
            draw_placeholder = ImageDraw.Draw(placeholder)
            draw_placeholder.text((10, 40), "No instruction\nimage", fill='black')
            canvas.paste(placeholder, (x, y))

        # For each camera view
        for view_idx, view in enumerate(['front', 'back', 'bottom']):
            if view in result.views and result.views[view]:
                try:
                    img = Image.open(result.views[view])
                    img = img.resize((img_size, img_size))

                    # Highlight the view that was used for prediction
                    if view == result.view_used:
                        overlay = Image.new('RGB', (img_size, img_size), 'red')
                        mask = Image.new('L', (img_size, img_size), 0)
                        for i in range(3):  # 3 pixel border
                            mask.paste(255, (i, i, img_size-i, img_size-i))
                        canvas.paste(overlay, (x + (view_idx + 1) * (img_size + 10), y), mask)

                    canvas.paste(img, (x + (view_idx + 2) * (img_size + 10), y))
                except Exception as e:
                    print(f"Error processing image {result.views[view]}: {str(e)}")

        # Add predicted class instruction manual image at the end
        pred_instruction_path = f"../lego-part-library/instruction/{result.pred_class}.png"
        try:
            pred_instruction_img = Image.open(pred_instruction_path)
            pred_instruction_img = pred_instruction_img.resize((img_size, img_size))
            canvas.paste(pred_instruction_img, (x + (img_size + 10), y))
        except Exception as e:
            print(f"Error loading predicted class instruction image for {result.pred_class}: {str(e)}")
            # Create a placeholder for missing instruction image
            placeholder = Image.new('RGB', (img_size, img_size), 'lightgray')
            draw_placeholder = ImageDraw.Draw(placeholder)
            draw_placeholder.text((10, 40), "No instruction\nimage", fill='black')
            canvas.paste(placeholder, (x + (img_size + 10), y))

        # Add text below images
        text = f"True: {result.true_class} Pred: {result.pred_class} ({result.confidence:.2f})"
        draw.text((x, y + img_size + 5), text, fill='black')

    # Save the visualization
    canvas.save(f'error_viz_{strategy_name}.png')

# 447 part nums we can classify today
part_nums = ['10197', '10288', '10314', '11090', '11211', '11212', '11213', '11214', '11215', '11272', '11458', '11476', '11477', '11478', '12939', '13547', '13548', '14417', '14419', '14704', '14707', '14716', '14720', '14769', '15068', '15070', '15092', '15100', '15332', '15395', '15397', '15460', '15461', '15470', '15535', '15672', '15706', '17485', '18649', '18651', '18653', '18674', '18677', '18969', '20482', '21229', '22385', '22388', '22390', '22391', '22484', '22885', '22888', '22889', '22890', '22961', '2339', '2357', '2362a', '23969', '24014', '24122', '2412a', '2419', '2420', '24246', '24299', '2431', '24316', '2436', '24375', '2441', '2445', '2450', '2453a', '2454a', '2456', '2458', '2460', '2476a', '2486', '24866', '25269', '2529', '2540', '2555', '25893', '26047', '2639', '2654', '26601', '26604', '27255', '27262', '27266', '2730', '2736', '27940', '2853', '2854', '2877', '2904', '2921', '2926', '30000', '3001', '3002', '3003', '3004', '30044', '30046', '3005', '3007', '3008', '3009', '30099', '3010', '30136', '30157', '30165', '30179', '3020', '3021', '3022', '3023', '30237a', '3024', '3028', '3031', '3032', '3033', '3034', '3035', '30357', '30361a', '30363', '30367b', '3037', '3038', '3039', '3040a', '30414', '3045', '30503', '30553', '30565', '3062b', '3063b', '3068a', '3069a', '3070a', '3185', '32000', '32002', '32013', '32014', '32015', '32016', '32017', '32028', '32034', '32039', '32054', '32056', '32059', '32064a', '32073', '32123a', '32124', '32126', '32140', '32184', '32187', '32192', '32198', '32209', '32250', '32278', '32291', '32316', '32348', '32449', '3245a', '32523', '32524', '32526', '32529', '32530', '32557', '32828', '32932', '3298', '3308', '33291', '33299a', '33909', '3403', '3455', '3460', '35044', '3622', '3623', '3633', '3639', '3640', '3647', '3648a', '3659', '3660', '3665', '3666', '3673', '3676', '3679', '3680', '3684', '36840', '36841', '3700', '3701', '3704', '3705', '3706', '3707', '3710', '3713', '3747a', '3749', '3794a', '3795', '3823', '3832', '3854', '3873', '3895', '3941', '3942', '3957a', '3958', '3960', '39739', '4032a', '40490', '4070', '4081b', '4083', '4085a', '40902', '4132', '41530', '41532', '4162', '41677', '41678', '41682', '41740', '41747', '41748', '4176', '41768', '41769', '41770', '4185', '42003', '42022', '42023', '4215a', '4216', '4218', '42446', '4274', '4282', '4286', '4287a', '43337', '43708', '43712', '43713', '43719', '43898', '44126', '44568', '4460b', '44728', '4477', '44809', '44874', '4488', '4490', '4510', '4519', '45590', '45677', '4589', '4600', '4727', '4733', '47397', '47398', '4740', '4742', '47456', '47753', '47755', '47905', '47994', '48092', '48169', '48171', '48336', '4855', '4864a', '4865', '4871', '48723', '48729a', '48933', '48989', '49668', '50304', '50305', '50373', '50950', '51739', '52031', '52501', '53923', '54200', '54383', '54384', '54661', '55013', '55981', '57518', '57519', '57520', '57585', '57895', '57908', '57909a', '57910', '58090', '58176', '59443', '60032', '6005', '6014', '6015', '6020', '60470a', '60471', '60474', '60475a', '60476', '60477', '60478', '60479', '60481', '60483', '60484', '60592', '60593', '60594', '60599', '6060', '60616', '60621', '60623', '6081', '6091', '61070', '61071', '6111', '61252', '61409', '6141', '6157', '61678', '6182', '6215', '6222', '6231', '6232', '6233', '62462', '63864', '63868', '63869', '64225', '64391', '64393', '64681', '64683', '64712', '6536', '6541', '6553', '6558', '6587', '6628', '6632', '6636', '72454', '74261', '84954', '85861', '85943', '85970', '85984', '87079', '87081', '87083', '87087', '87580', '87609', '87620', '88292', '88646', '88930', '90195', '90202', '90609', '90611', '90630', '92092', '92280', '92579', '92582', '92583', '92589', '92907', '92947', '93273', '93274', '93606', '94161', '98100', '98138', '98262', '98283', '99008', '99021', '99206', '99207', '99773', '99780', '99781']

# evaluator = FusionEvaluator("lego-classify-05-447-fixed-num.pt")
# def compensate(part_num):
#     return rebrickable_to_canonical(part_num)
# evaluator = FusionEvaluator("lego-classify-03-447x.pt")
# def compensate(part_num):
#     return model_03_to_canonical(part_num)
evaluator = FusionEvaluator("lego-classify-06-447-fixed-nums-2.pt")
def compensate(part_num):
    return part_num

groups = group_images('src/v1/')
examples = [(part_num, views) for part_num in groups for _ts, views in groups[part_num]]
examples = [example for example in examples if example[0] in part_nums]
print("Num examples:", len(examples))
print("Num classes:", len(set([example[0] for example in examples])))

# Run evaluations

metrics, results = evaluator.fuse_topk_confidence(examples, k=5)
create_error_visualization(results, "topk_confidence")
print("TopK Confidence:", metrics)

metrics, results = evaluator.fuse_ensemble_voting(examples)
create_error_visualization(results, "ensemble_voting")
print("Ensemble voting:", metrics)

metrics, results = evaluator.fuse_sequential_confidence(examples)
create_error_visualization(results, "sequential_confidence")
print("Sequential Confidence:", metrics)

metrics, results = evaluator.evaluate_front_only(examples)
create_error_visualization(results, "front_only")
print("Front Only:", metrics)

metrics, results = evaluator.evaluate_first_available(examples)
create_error_visualization(results, "first_available")
print("First Available:", metrics)

metrics, results = evaluator.fuse_max_confidence(examples)
create_error_visualization(results, "max_confidence")
print("Max Confidence:", metrics)

# Results
#   Num examples: 303
#   Num classes: 47
#   Top3 Confidence: {'accuracy': 0.9042904290429042, 'failed_predictions': 0, 'total': 303, 'view_usage': {'back': 149, 'front': 99, 'bottom': 55}}
#   Top5 same as Top3
#   Ensemble voting: {'accuracy': 0.8844884488448845, 'failed_predictions': 0, 'total': 303, 'view_usage': {'back': 148, 'front': 96, 'bottom': 59}}
#   Sequential Confidence: {'accuracy': 0.8778877887788779, 'failed_predictions': 0, 'total': 303, 'view_usage': {'back': 66, 'front': 176, 'bottom': 61}}
#   Front Only: {'accuracy': 0.7623762376237624, 'failed_predictions': 11, 'total': 303}
#   First Available: {'accuracy': 0.7887788778877888, 'failed_predictions': 0, 'total': 303, 'view_usage': {'front': 292, 'bottom': 11}}
#   Max Confidence: {'accuracy': 0.8778877887788779, 'failed_predictions': 0, 'total': 303, 'view_usage': {'back': 146, 'front': 97, 'bottom': 60}}
