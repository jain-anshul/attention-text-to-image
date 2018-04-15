from miscc.config import cfg_from_file, cfg
from datasets import TextDataset
import argparse
import pprint
import random
import numpy as np
import datetime
import dateutil.tz
import time

import torch
import torchvision.transforms as transforms

from trainer import condGANTrainer as trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train the AttnGAN network")
    parser.add_argument('--cfg', dest="cfg_file",
                        help="optional config file (birds default)",
                        default="cfg/birds_attn2.yml", type=str)
    parser.add_argument('--gpu', dest="gpu_id", type=int, default=-1)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    """
    Merge cfg and data obtained from arguments
    """
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        """
        To get the same answer??
        """
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed(args.manualSeed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
                 (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    """
    Use??
    """
    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        split_dir = 'test'

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))

    """
    Why are we rescaling?
    """
    image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = TextDataset(cfg.DATA_DIR, split_dir,
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)

    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS)
    )

    # Define models and go to train/evaluate
    algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword)

    start_t = time.time()
    if cfg.TRAIN.FLAG:
        algo.train()
    # else:
    #     '''generate images from pre-extracted embeddings'''
    #     if cfg.B_VALIDATION:
    #         algo.sampling(split_dir)  # generate images for the whole valid dataset
    #     else:
    #         gen_example(dataset.wordtoix, algo)
    end_t = time.time()
    print('Total time for training:', end_t - start_t)
