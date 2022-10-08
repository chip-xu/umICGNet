import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import dataloader
from utils.metrics import evaluate, Metrics
from args import args, device, generate_model, num_workers
from utils.loss import Deep_Loss
import torch.nn.functional as F


def train():
    model = generate_model()

    # load data
    train_data = getattr(dataloader, args.dataset)(args.root, args.train_data_dir, mode='train')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    valid_data = getattr(dataloader, args.dataset)(args.root, args.valid_data_dir, mode='valid')
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=num_workers)
    val_total_batch = len(valid_data)

    # load optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)

    print('Start training')
    print('---------------------------------\n')
    for epoch in range(args.Epoch):
        print('------ Epoch', epoch + 1)
        model.train()   # train
        bar = tqdm(enumerate(train_loader), total=len(train_data))
        
        if epoch+1 in [60, 120]:
            optimizer.param_groups[0]['lr'] *= 0.5
                
        for i, (data, file_name) in bar:
            img = data['image'].to(device)
            gt = data['label'].to(device)

            output = model(img)
            loss = Deep_Loss(output, gt)
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step() 
            bar.set_postfix_str('loss: %.4s' % loss.item()) 

        model.eval()  # valid
        metrics = Metrics(['recall', 'precision', 'F1', 'IoU_mean', 'ACC_overall'])
        with torch.no_grad():
            bar = tqdm(enumerate(valid_loader), total=len(valid_data))
            for i, (data, file_name) in bar:
                img = data['image'].to(device)
                gt = data['label'].to(device)

                predict = model(img)
                predict = predict[0]
    
                _recall, _precision, _F1, _IoU_mean, _ACC_overall = evaluate(predict, gt)
                metrics.update(recall=_recall, precision=_precision, F1=_F1, IoU_mean=_IoU_mean,
                               ACC_overall=_ACC_overall)
        metrics_result = metrics.mean(val_total_batch)
        print('\n')
        print('recall: %.4f, precision: %.4f, Dice: %.4f, mIoU: %.4f,  ACC: %.4f,'
              % (metrics_result['recall'], metrics_result['precision'], metrics_result['F1'],
                 metrics_result['IoU_mean'], metrics_result['ACC_overall']))
        print('lr: {}'.format(optimizer.param_groups[0]['lr']))

        if (epoch + 1) % args.ckpt_period == 0:
            torch.save(model.state_dict(), './result' + "/ck_{}.pth".format(epoch + 1))


if __name__ == '__main__':
    train()
    print('Done')

