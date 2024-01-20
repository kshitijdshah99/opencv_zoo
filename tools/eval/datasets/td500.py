import os
import tqdm
import pickle
import numpy as np
from scipy.io import loadmat
import cv2 as cv

def get_gt_boxes_text(gt_dir):
    gt_file = os.path.join(gt_dir, 'gt.mat')

    try:
        gt_mat = loadmat(gt_file)
    except FileNotFoundError:
        print(f"Warning: 'gt.mat' not found in {gt_dir}. Returning empty list.")
        return []

    gt_boxes = []
    for i in range(len(gt_mat['gt'][0])):
        img_gt = gt_mat['gt'][0][i][0][0][0][0]
        boxes = []
        for j in range(len(img_gt)):
            box = img_gt[j][0][0][0][0][0]
            boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
        gt_boxes.append(boxes)

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

def evaluation_text(pred, gt_path, iou_thresh=0.5):
    norm_score_text(pred)

    # Modify to handle missing 'gt.mat' files
    gt_boxes_list = get_gt_boxes_text(gt_path)
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


# The rest of the code remains unchanged
class TD500:
    def __init__(self, root, split='test'):
        self.aps = []
        self.msra_root = root
        self._split = split

        self.msra_img_paths = {
            'test': os.path.join(self.msra_root, 'MSRA-TD500', 'test')
        }

        self.gt_path = os.path.join(self.msra_root, 'gt')

        self.img_list, self.num_img = self.load_list()

    def load_list(self):
        n_imgs = 0
        flist = []

        img_path = self.msra_img_paths[self._split]
        gt_path = self.gt_path

        for i in range(1, 301):  # Assuming there are 300 images in the MSRA TD500 dataset
            img_file = os.path.join(img_path, '{}.jpg'.format(i))
            gt_file = os.path.join(gt_path, '{}.mat'.format(i))
            if os.path.exists(img_file) and os.path.exists(gt_file):
                flist.append((img_file, gt_file))
                n_imgs += 1

        return flist, n_imgs

    def __getitem__(self, index):
        img_file, gt_file = self.img_list[index]
        img = cv.imread(img_file)
        gt_boxes = get_gt_boxes_text(os.path.dirname(gt_file))
        return gt_boxes, img

    def eval(self, model):
        results_list = dict()
        pbar = tqdm.tqdm(self)
        pbar.set_description_str("Evaluating {} with {} test set".format(model.name, self._split))
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

        self.aps = evaluation_text(results_list, os.path.join(self.msra_root, 'gt'))

    def print_result(self):
        print("==================== Results ====================")
        print("Text Detection AP: {}".format(np.mean(self.aps)))
        print("=================================================")
