import os
import tqdm
import pickle
import numpy as np
from scipy.io import loadmat
import cv2 as cv

# def get_gt_boxes_text(img_dir):
#     print(img_dir)
#     img_files = [f for f in os.listdir(img_dir) if f.endswith('.JPG')]
#     gt_boxes = []

#     for img_file in img_files:
#         img_path = os.path.join(img_dir, img_file)
#         gt_path = os.path.splitext(img_path)[0] + '.gt'
        
#         print("Checking for ground truth file:", gt_path)  # Debug print

#         if os.path.exists(gt_path):
#             with open(gt_path, 'r') as file:
#                 lines = file.readlines()
#                 boxes = []

#                 for line in lines:
#                     parts = line.strip().split(',')
#                     box = [float(part) for part in parts]
#                     boxes.append(box)

#                 gt_boxes.append(boxes)
#         else:
#             print("Ground truth file not found:", gt_path)  # Debug print

#     return gt_boxes

def get_gt_boxes_text(img_dir):
    print(img_dir)
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.JPG')]
    gt_boxes = []

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        gt_path = os.path.splitext(img_path)[0] + '.gt'

        print("Checking for ground truth file:", gt_path)  # Debug print

        if os.path.exists(gt_path):
            with open(gt_path, 'r') as file:
                lines = file.readlines()
                boxes = []

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        box = [float(part) for part in parts[:7]]
                        boxes.append(box)
                        print(f"Values: {box}")

                    else:
                        print(f"Ignoring invalid line in {gt_path}: {line}")

                gt_boxes.append(boxes)
        else:
            print("Ground truth file not found:", gt_path)  # Debug print

    return gt_boxes



def norm_score_text(pred):
    max_score = 0
    min_score = 1

    for _, v in pred.items():
        if len(v) == 0:
            continue
        _min = np.min(v[:, -1])
        _max = np.max(v[:, -1])
        max_score = max(_max, max_score)
        min_score = min(_min, min_score)

    diff = max_score - min_score
    for _, v in pred.items():
        if len(v) == 0:
            continue
        v[:, -1] = (v[:, -1] - min_score) / diff

def evaluation_text(pred, img_dir, iou_thresh=0.5):
    norm_score_text(pred)

    gt_boxes_list = get_gt_boxes_text(img_dir)
    if not gt_boxes_list:
        print("No ground truth data available. Skipping evaluation.")
        return []

    thresh_num = 1000
    aps = []

    for gt_boxes in gt_boxes_list:
        count_text = 0
        pr_curve = np.zeros((thresh_num, 2)).astype('float')
        pbar = tqdm.tqdm(range(len(gt_boxes)))
        for i in pbar:
            pbar.set_description('Processing GT #{}'.format(i + 1))

            img_gt_boxes = gt_boxes[i]
            pred_info = pred.get(str(i + 1), np.array([[10, 10, 20, 20, 0.002]]))

            ignore = np.zeros(len(img_gt_boxes))
            pred_recall, proposal_list = image_eval_text(pred_info, np.array(img_gt_boxes), ignore, iou_thresh)

            _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)

            pr_curve += _img_pr_info
            count_text += len(img_gt_boxes)

        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_text)

        propose = pr_curve[:, 0]
        recall = pr_curve[:, 1]

        ap = voc_ap(recall, propose)
        aps.append(ap)

    return aps

# def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
#     # Sample data (replace with actual data from your implementation)
#     # Assuming pred_info is a list of dictionaries with 'precision' and 'recall' keys
#     precision_values = np.random.rand(thresh_num).tolist()  # Replace with actual precision values
#     recall_values = np.random.rand(thresh_num).tolist()  # Replace with actual recall values

#     # Create a new dictionary to store the PR information
#     pr_info = {
#         'precision': precision_values,
#         'recall': recall_values,
#     }

#     # Calculate other relevant metrics based on your implementation
#     true_positives = np.random.randint(0, 10, thresh_num).tolist()  # Replace with actual true positive values
#     false_positives = np.random.randint(0, 10, thresh_num).tolist()  # Replace with actual false positive values
#     false_negatives = np.random.randint(0, 10, thresh_num).tolist()  # Replace with actual false negative values

#     # Add additional metrics to the pr_info dictionary if needed
#     pr_info['true_positives'] = true_positives
#     pr_info['false_positives'] = false_positives
#     pr_info['false_negatives'] = false_negatives

#     # Update image-level information based on proposals and predicted recall
#     image_recall = len(proposal_list) / pred_recall
#     image_precision = len(proposal_list) / (len(proposal_list) + false_positives[0])

#     pr_info['image_recall'] = image_recall
#     pr_info['image_precision'] = image_precision

#     # Append the pr_info dictionary to the pred_info list
#     pred_info.append(pr_info)

#     return pred_info



def image_eval_text(pred_info, gt_boxes, ignore, iou_thresh):
    """
    Evaluate text detection at the image level.

    Parameters:
    - pred_info (numpy.ndarray): Model predictions for the image.
    - gt_boxes (numpy.ndarray): Ground truth bounding boxes for the image.
    - ignore (numpy.ndarray): Ignore flags for each ground truth box.
    - iou_thresh (float): IoU threshold for matching predictions to ground truth.

    Returns:
    - pred_recall (float): Predicted recall for the image.
    - proposal_list (list): List of proposals for the image.
    """
    # Placeholder implementation, replace with actual logic
    # You should modify this based on the output format of your PPOCRDet model
    # and the evaluation criteria of the TD500 dataset

    # For example, assuming pred_info has columns [x1, y1, x2, y2, confidence]
    pred_boxes = pred_info[:, :4]
    pred_confidences = pred_info[:, 4]

    # Perform non-maximum suppression (NMS) to filter out redundant predictions
    # NMS implementation depends on your PPOCRDet model's output format
    selected_indices = nms(pred_boxes, pred_confidences, iou_thresh)

    # Get the selected predictions after NMS
    selected_pred_boxes = pred_boxes[selected_indices]
    
    # Evaluate the selected predictions against ground truth boxes
    true_positives = 0
    false_positives = 0

    for pred_box in selected_pred_boxes:
        iou_scores = calculate_iou(gt_boxes, pred_box)
        max_iou = np.max(iou_scores)

        if max_iou >= iou_thresh:
            true_positives += 1
        else:
            false_positives += 1

    # Calculate predicted recall
    pred_recall = true_positives / (true_positives + np.sum(ignore))

    # Create a list of proposals for the image (optional, depending on your needs)
    proposal_list = selected_pred_boxes.tolist()

    return pred_recall, proposal_list


# def nms(boxes, scores, iou_thresh):
#     """
#     Perform non-maximum suppression (NMS) on boxes based on scores.

#     Parameters:
#     - boxes (numpy.ndarray): Bounding boxes (format: [x1, y1, x2, y2]).
#     - scores (numpy.ndarray): Confidence scores for each box.
#     - iou_thresh (float): IoU threshold for NMS.

#     Returns:
#     - selected_indices (numpy.ndarray): Indices of selected boxes after NMS.
#     """
#     # Placeholder implementation, replace with actual NMS logic
#     # You should use a proper NMS algorithm compatible with your PPOCRDet model

#     # Sorting indices based on scores in descending order
#     sorted_indices = np.argsort(scores)[::-1]

#     selected_indices = []

#     for i in sorted_indices:
#         if len(selected_indices) == 0:
#             selected_indices.append(i)
#         else:
#             iou_scores = calculate_iou(boxes[selected_indices], boxes[i])
#             if np.max(iou_scores) < iou_thresh:
#                 selected_indices.append(i)

#     return np.array(selected_indices)


# def calculate_iou(boxes1, boxes2):
#     if len(boxes1.shape) == 1:
#         boxes1 = boxes1.reshape(1, -1)

#     if len(boxes2.shape) == 1:
#         boxes2 = boxes2.reshape(1, -1)

#     intersection_x1 = np.maximum(boxes1[:, 0], boxes2[0, 0])
#     intersection_y1 = np.maximum(boxes1[:, 1], boxes2[0, 1])
#     intersection_x2 = np.minimum(boxes1[:, 2], boxes2[0, 2])
#     intersection_y2 = np.minimum(boxes1[:, 3], boxes2[0, 3])

#     intersection_area = np.maximum(0, intersection_x2 - intersection_x1 + 1) * np.maximum(0, intersection_y2 - intersection_y1 + 1)

#     area_boxes1 = (boxes1[:, 2] - boxes1[:, 0] + 1) * (boxes1[:, 3] - boxes1[:, 1] + 1)
#     area_boxes2 = (boxes2[0, 2] - boxes2[0, 0] + 1) * (boxes2[0, 3] - boxes2[0, 1] + 1)

#     iou_scores = intersection_area / (area_boxes1 + area_boxes2 - intersection_area)

#     return iou_scores





# def dataset_pr_info(thresh_num, pr_curve, count_text):
#     # Sample data (replace with actual data from your evaluation)
#     # Assuming pr_curve is a dictionary with 'precision', 'recall', and 'threshold' keys
#     precision_values = np.random.rand(thresh_num)  # Replace with actual precision values
#     recall_values = np.random.rand(thresh_num)  # Replace with actual recall values
#     threshold_values = np.linspace(0, 1, thresh_num)  # Replace with actual threshold values

#     # Update PR curve information
#     pr_curve['precision'] = precision_values
#     pr_curve['recall'] = recall_values
#     pr_curve['threshold'] = threshold_values

#     # Calculate other relevant metrics based on your dataset and evaluation
#     true_positives = count_text * precision_values  # Replace with actual calculation
#     false_positives = np.random.randint(0, 10, thresh_num)  # Replace with actual false positive values
#     false_negatives = np.random.randint(0, 10, thresh_num)  # Replace with actual false negative values

#     # Update the pr_curve dictionary with additional metrics if needed
#     pr_curve['true_positives'] = true_positives
#     pr_curve['false_positives'] = false_positives
#     pr_curve['false_negatives'] = false_negatives

#     return pr_curve



# def voc_ap(recall, propose):
#     # Convert lists to numpy arrays
#     recall = np.array(recall)
#     propose = np.array(propose)

#     # Ensure both arrays are non-empty and of the same length
#     if recall.size == 0 or propose.size == 0 or recall.size != propose.size:
#         raise ValueError("Input arrays must be non-empty and of the same length.")

#     # Sorting by propose values in descending order
#     sorted_data = sorted(zip(recall, propose), key=lambda x: x[1], reverse=True)

#     # Initializing variables
#     true_positives = 0
#     false_positives = 0
#     precision_values = []
#     recall_values = []

#     # Calculating precision and recall values
#     for recall_val, propose_val in sorted_data:
#         recall_values.append(recall_val)
#         if propose_val == 1:
#             true_positives += 1
#         else:
#             false_positives += 1
        
#         precision = true_positives / (true_positives + false_positives)
#         print(precision)
#         precision_values.append(precision)

#     # Calculating Average Precision using the trapezoidal rule
#     ap = 0
#     for i in range(len(precision_values) - 1):
#         ap += (recall_values[i + 1] - recall_values[i]) * precision_values[i + 1]
    
#     return ap




# The rest of the code remains unchanged
class TD500:
    def __init__(self, root, split='test'):
        self.aps = []
        self.msra_root = root
        self._split = split

        self.msra_img_path = os.path.join(self.msra_root, 'MSRA-TD500', self._split)

        self.img_list, self.num_img = self.load_list()

    def load_list(self):
        n_imgs = 0
        flist = []

        img_path = self.msra_img_path

        if os.path.exists(img_path):
            img_files = [f for f in os.listdir(img_path) if f.endswith('.jpg')]

            for img_file in img_files:
                img_file_path = os.path.join(img_path, img_file)
                flist.append(img_file_path)
                n_imgs += 1
        else:
            print(f"No such directory: {img_path}")

        return flist, n_imgs

    def __getitem__(self, index):
        img_file = self.img_list[index]
        img = cv.imread(img_file)
        gt_file = os.path.join(self.msra_img_path, os.path.splitext(os.path.basename(img_file))[0] + '.gt')
        gt_boxes = get_gt_boxes_text(gt_file)
        return gt_boxes, img

    def eval(self, model):
        results_list = dict()
        pbar = tqdm.tqdm(self)
        pbar.set_description_str("Evaluating {} with {} set".format(model.name, self._split))
        for gt_boxes, img in pbar:
            img_shape = [img.shape[1], img.shape[0]]
            model.setInputSize(img_shape)
            det = model.infer(img)

            if not results_list.get(str(len(results_list) + 1)):
                results_list[str(len(results_list) + 1)] = dict()

            if det is None:
                det = np.array([[10, 10, 20, 20, 0.002]])
            else:
                det = np.append(np.around(det[:, :4], 1), np.around(det[:, -1], 3).reshape(-1, 1), axis=1)

            results_list[str(len(results_list))][str(len(results_list) + 1)] = det

        self.aps = evaluation_text(results_list, self.msra_img_path)

    def print_result(self):
        print("==================== Results ====================")
        print("Text Detection AP: {}".format(np.mean(self.aps)))
        print("=================================================")
