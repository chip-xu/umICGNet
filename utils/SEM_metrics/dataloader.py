from torch.utils import data
import os
from PIL import Image


class EvalDataset(data.Dataset):
    def __init__(self, pred_root, gt_root):
        lst_pred = sorted(os.listdir(pred_root))
        lst_gt = sorted(os.listdir(gt_root))
        lst = []
        for name in lst_gt:
            if name in lst_pred:
                lst.append(name)

        self.lst = lst
        self.pred_path = list(map(lambda x: os.path.join(pred_root, x), lst))
        self.gt_path = list(map(lambda x: os.path.join(gt_root, x), lst))

    def __getitem__(self, item):
        pred = Image.open(self.pred_path[item]).convert('L')
        gt = Image.open(self.gt_path[item]).convert('L')
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)
        img_name = self.lst[item]

        return pred, gt, img_name

    def __len__(self):
        return len(self.pred_path)
