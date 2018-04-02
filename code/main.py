from miscc.config import cfg_from_file
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train the AttnGAN network")
    parser.add_argument('--cfg', dest="cfg_file",
                        help="optional config file (birds default)",
                        default="cfg/birds/attn2.yml", type=str)
    parser.add_argument('--gpu', dest="gpu_id", type=int, default=-1)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
