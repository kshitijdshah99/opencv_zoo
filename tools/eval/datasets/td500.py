import os
import tqdm
import pickle
import numpy as np
from scipy.io import loadmat
import cv2 as cv

def get_gt_boxes_text(img_dir):
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    gt_boxes = []

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        gt_path = os.path.splitext(img_path)[0] + '.gt'
        
        if os.path.exists(gt_path):
            with open(gt_path, 'r') as file:
                lines = file.readlines()
                boxes = []

                for line in lines:
                    parts = line.strip().split(',')
                    box = [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])]
                    boxes.append(box)

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
