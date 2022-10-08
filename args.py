import torch
import argparse
import models
import os


parse = argparse.ArgumentParser(description='PyTorch Polyp Segmentation')
"-------------------data option--------------------------"
parse.add_argument('--root', type=str, default='./datasets/EndoScene', help='Cross_data, EndoScene, Kvasir-SEG')
parse.add_argument('--dataset', type=str, default='Polyp_Dataset')
parse.add_argument('--train_data_dir', type=str, default='train')
parse.add_argument('--valid_data_dir', type=str, default='valid')
parse.add_argument('--test_data_dir', type=str, default='test', help='test, test_ColonDB, test_CVC300, test_ETIS')

"-------------------training option-----------------------"
parse.add_argument('--Epoch', type=int, default=150)
parse.add_argument('--batch_size', type=int, default=32)
parse.add_argument('--load_ckpt', type=str, default=None)
parse.add_argument('--model', type=str, default='umICGNet')
parse.add_argument('--ckpt_period', type=int, default=5)

"-------------------optimizer option-----------------------"
parse.add_argument('--base_lr', type=float, default=1e-4)
parse.add_argument('--weight_decay', type=float, default=5e-4)
parse.add_argument('--momentum', type=float, default=0.9)
parse.add_argument('--power', type=float, default=0.9)
parse.add_argument('--num_classes', type=int, default=1)
args = parse.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
# print('Using {} dataloader workers every process'.format(num_workers))


def generate_model():
    net = getattr(models, args.model)(args.num_classes)
    net.to(device)

    if args.load_ckpt is not None:
        net.load_state_dict(torch.load('./checkpoints/EndoScene.pth'))
        print('Done')

    return net
