import torch
from args import args, device, generate_model, num_workers
from utils.metrics import evaluate, Metrics
import dataloader
from torch.utils.data import DataLoader
import numpy as np
from utils import helpers
import os
import time
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F


def test():
    print('loading data......')
    model = generate_model()
    test_data = getattr(dataloader, args.dataset)(args.root, args.test_data_dir, mode='test')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=num_workers)
    total_batch = len(test_data)

    model.eval()
    metrics = Metrics(['recall', 'precision', 'F1', 'IoU_mean', 'ACC_overall'])
    time_taken = []
    with torch.no_grad():
        for val_batch, (data, file_name) in enumerate(test_loader, 1):
            img = data['image'].to(device)
            gt = data['label'].to(device)
            file_name = file_name[0].split('.')[0]

            start_time = time.time()
            predict1 = model(img)
            predict = predict1[0]
            end_time = time.time()
            total_time = end_time - start_time
            time_taken.append(total_time)

            out2 = F.interpolate(predict1[1], size=predict.size()[2:], mode='bilinear', align_corners=True)
            out3 = F.interpolate(predict1[2], size=predict.size()[2:], mode='bilinear', align_corners=True)
            out4 = F.interpolate(predict1[3], size=predict.size()[2:], mode='bilinear', align_corners=True)
            out5 = F.interpolate(predict1[4], size=predict.size()[2:], mode='bilinear', align_corners=True)
            Avg = (predict + out5 + out4 + out3 + out2) / 5
            uncertain_map = -1.0 * torch.sum(Avg * torch.log(Avg + 1e-6), dim=1, keepdim=True)
            u_png1 = np.array(uncertain_map.cpu().squeeze())
            u_png2 = np.expand_dims(u_png1, 2)
            u_png3 = helpers.array_to_img(u_png2)
            plt.imsave(os.path.join('F:/umICGNet/predict/u_mask', file_name + '.png'), u_png3)

            png1 = np.array(predict.cpu().squeeze())
            png2 = np.expand_dims(png1, 2)
            png3 = helpers.array_to_img(png2)
            png3.save(os.path.join('F:/umICGNet/predict/mask', file_name + '.png'))

            _recall, _precision, _F1, _IoU_mean, _ACC_overall = evaluate(predict, gt)
            metrics.update(recall=_recall, precision=_precision, F1=_F1, IoU_mean=_IoU_mean,
                           ACC_overall=_ACC_overall)

    metrics_result = metrics.mean(total_batch)
    mean_time_taken = np.mean(time_taken)
    mean_fps = 1 / mean_time_taken
    print("Mean FPS: ", mean_fps)
    print("Test Result:")
    print('recall: %.4f, precision: %.4f, Dice: %.4f, mIoU: %.4f,  ACC: %.4f,'
          % (metrics_result['recall'], metrics_result['precision'], metrics_result['F1'],
             metrics_result['IoU_mean'], metrics_result['ACC_overall']))


if __name__ == '__main__':
    test()
