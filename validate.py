from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import Counter
from PIL import Image

from lib.grouper import group_images

class FusionEvaluator:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict_single_image(self, image_path: Optional[str]) -> Tuple[Optional[str], float]:
        """Predict class and confidence for a single image"""
        if image_path is None or not Path(image_path).exists():
            return None, 0.0

        try:
            image = Image.open(image_path)
            results = self.model.predict(image, verbose=False)
            pred_class = results[0].names[results[0].probs.top1]
            confidence = results[0].probs.top1conf.item()
            return pred_class, confidence
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None, 0.0

    def evaluate_front_only(self, examples: List[Tuple[str, Dict[str, str]]]) -> Dict:
        """Baseline: Use only front view predictions"""
        correct = 0
        total = 0
        failed_predictions = 0

        for true_class, views in examples:
            pred_class, confidence = self.predict_single_image(views.get('front'))

            if pred_class is None:
                failed_predictions += 1
                continue

            total += 1
            if pred_class == true_class:
                correct += 1

        return {
            'accuracy': correct / total if total > 0 else 0,
            'failed_predictions': failed_predictions,
            'total': len(examples)
        }

    def evaluate_first_available(self, examples: List[Tuple[str, Dict[str, str]]]) -> Dict:
        """Use first available view (front -> back -> bottom)"""
        correct = 0
        total = 0
        failed_predictions = 0
        view_used_counts = Counter()

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
                        break
            else:
                failed_predictions += 1

        return {
            'accuracy': correct / total if total > 0 else 0,
            'failed_predictions': failed_predictions,
            'total': len(examples),
            'view_usage': dict(view_used_counts)
        }

    def get_all_predictions(self, views: Dict[str, str]) -> List[Tuple[str, str, float]]:
        """Get predictions for all available views"""
        predictions = []
        for view_name, path in views.items():
            pred_class, confidence = self.predict_single_image(path)
            if pred_class is not None:
                predictions.append((view_name, pred_class, confidence))
        return predictions

    def fuse_max_confidence(self, examples: List[Tuple[str, Dict[str, str]]]) -> Dict:
        """Use prediction with highest confidence across all views"""
        correct = 0
        total = 0
        failed_predictions = 0
        view_used_counts = Counter()

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

        return {
            'accuracy': correct / total if total > 0 else 0,
            'failed_predictions': failed_predictions,
            'total': len(examples),
            'view_usage': dict(view_used_counts)
        }

    def fuse_majority_vote(self, examples: List[Tuple[str, Dict[str, str]]]) -> Dict:
        """Use majority vote across all views, breaking ties by confidence"""
        correct = 0
        total = 0
        failed_predictions = 0
        view_contribution_counts = Counter()

        for true_class, views in examples:
            predictions = self.get_all_predictions(views)

            if not predictions:
                failed_predictions += 1
                continue

            # Count votes for each class
            class_votes = Counter(pred[1] for pred in predictions)
            max_votes = max(class_votes.values())

            # Get classes with maximum votes
            top_classes = [cls for cls, votes in class_votes.items() if votes == max_votes]

            if len(top_classes) == 1:
                pred_class = top_classes[0]
                # Track which views contributed to the majority
                for view_name, p_class, _ in predictions:
                    if p_class == pred_class:
                        view_contribution_counts[view_name] += 1
            else:
                # Break tie using confidence
                view_name, pred_class, _ = max(
                    (p for p in predictions if p[1] in top_classes),
                    key=lambda x: x[2]
                )
                view_contribution_counts[view_name] += 1

            total += 1
            if pred_class == true_class:
                correct += 1

        return {
            'accuracy': correct / total if total > 0 else 0,
            'failed_predictions': failed_predictions,
            'total': len(examples),
            'view_contributions': dict(view_contribution_counts)
        }

# 447 part nums we can classify today
part_nums = ['10197', '10288', '10314', '11090', '11211', '11212', '11213', '11214', '11215', '11272', '11458', '11476', '11477', '11478', '12939', '13547', '13548', '14417', '14419', '14704', '14707', '14716', '14720', '14769', '15068', '15070', '15092', '15100', '15332', '15395', '15397', '15460', '15461', '15470', '15535', '15672', '15706', '17485', '18649', '18651', '18653', '18674', '18677', '18969', '20482', '21229', '22385', '22388', '22390', '22391', '22484', '22885', '22888', '22889', '22890', '22961', '2339', '2357', '2362a', '23969', '24014', '24122', '2412a', '2419', '2420', '24246', '24299', '2431', '24316', '2436', '24375', '2441', '2445', '2450', '2453a', '2454a', '2456', '2458', '2460', '2476a', '2486', '24866', '25269', '2529', '2540', '2555', '25893', '26047', '2639', '2654', '26601', '26604', '27255', '27262', '27266', '2730', '2736', '27940', '2853', '2854', '2877', '2904', '2921', '2926', '30000', '3001', '3002', '3003', '3004', '30044', '30046', '3005', '3007', '3008', '3009', '30099', '3010', '30136', '30157', '30165', '30179', '3020', '3021', '3022', '3023', '30237a', '3024', '3028', '3031', '3032', '3033', '3034', '3035', '30357', '30361a', '30363', '30367b', '3037', '3038', '3039', '3040a', '30414', '3045', '30503', '30553', '30565', '3062b', '3063b', '3068a', '3069a', '3070a', '3185', '32000', '32002', '32013', '32014', '32015', '32016', '32017', '32028', '32034', '32039', '32054', '32056', '32059', '32064a', '32073', '32123a', '32124', '32126', '32140', '32184', '32187', '32192', '32198', '32209', '32250', '32278', '32291', '32316', '32348', '32449', '3245a', '32523', '32524', '32526', '32529', '32530', '32557', '32828', '32932', '3298', '3308', '33291', '33299a', '33909', '3403', '3455', '3460', '35044', '3622', '3623', '3633', '3639', '3640', '3647', '3648a', '3659', '3660', '3665', '3666', '3673', '3676', '3679', '3680', '3684', '36840', '36841', '3700', '3701', '3704', '3705', '3706', '3707', '3710', '3713', '3747a', '3749', '3794a', '3795', '3823', '3832', '3854', '3873', '3895', '3941', '3942', '3957a', '3958', '3960', '39739', '4032a', '40490', '4070', '4081b', '4083', '4085a', '40902', '4132', '41530', '41532', '4162', '41677', '41678', '41682', '41740', '41747', '41748', '4176', '41768', '41769', '41770', '4185', '42003', '42022', '42023', '4215a', '4216', '4218', '42446', '4274', '4282', '4286', '4287a', '43337', '43708', '43712', '43713', '43719', '43898', '44126', '44568', '4460b', '44728', '4477', '44809', '44874', '4488', '4490', '4510', '4519', '45590', '45677', '4589', '4600', '4727', '4733', '47397', '47398', '4740', '4742', '47456', '47753', '47755', '47905', '47994', '48092', '48169', '48171', '48336', '4855', '4864a', '4865', '4871', '48723', '48729a', '48933', '48989', '49668', '50304', '50305', '50373', '50950', '51739', '52031', '52501', '53923', '54200', '54383', '54384', '54661', '55013', '55981', '57518', '57519', '57520', '57585', '57895', '57908', '57909a', '57910', '58090', '58176', '59443', '60032', '6005', '6014', '6015', '6020', '60470a', '60471', '60474', '60475a', '60476', '60477', '60478', '60479', '60481', '60483', '60484', '60592', '60593', '60594', '60599', '6060', '60616', '60621', '60623', '6081', '6091', '61070', '61071', '6111', '61252', '61409', '6141', '6157', '61678', '6182', '6215', '6222', '6231', '6232', '6233', '62462', '63864', '63868', '63869', '64225', '64391', '64393', '64681', '64683', '64712', '6536', '6541', '6553', '6558', '6587', '6628', '6632', '6636', '72454', '74261', '84954', '85861', '85943', '85970', '85984', '87079', '87081', '87083', '87087', '87580', '87609', '87620', '88292', '88646', '88930', '90195', '90202', '90609', '90611', '90630', '92092', '92280', '92579', '92582', '92583', '92589', '92907', '92947', '93273', '93274', '93606', '94161', '98100', '98138', '98262', '98283', '99008', '99021', '99206', '99207', '99773', '99780', '99781']
evaluator = FusionEvaluator("lego-classify-05-447-fixed-num.pt")

groups = group_images('src/v1/')
examples = [(part_num, views) for part_num in groups for _ts, views in groups[part_num]]
examples = [example for example in examples if example[0] in part_nums]
print("Num examples:", len(examples))
print("Num classes:", len(set([example[0] for example in examples]))
print(examples[1])

# Run evaluations
print("Front Only:", evaluator.evaluate_front_only(examples))
print("First Available:", evaluator.evaluate_first_available(examples))
print("Max Confidence:", evaluator.fuse_max_confidence(examples))
print("Majority Vote:", evaluator.fuse_majority_vote(examples))
