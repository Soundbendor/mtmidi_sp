import torch, optuna, pickle, numpy as np

import util.util_main as UM
from models.linearnnprobe import LinearNNProbe

from functools import partial
from distutils.util import strtobool
import os, sys time, argparse, tomllib

if __name__ == "__main__":
    #### arg parsing
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ds", "--dataset", type=str, default="polyrhythms", help="dataset")
    parser.add_argument("-ms", "--model_size", type=str, default="small", help="small/medium/large")
    parser.add_argument("-wdb", "--use_wandb", type=strtobool, default=True, help="sync to wandb")
    parser.add_argument("-cd", "--use_cuda", type=strtobool, default=True, help="use cuda")
    parser.add_argument("-ev", "--eval", type=strtobool, default=False, help="evalute on best performing params recorded")
    parser.add_argument("-ev", "--eval", type=strtobool, default=False, help="evalute on best performing params recorded")
    parser.add_argument("-sh", "--from_share", type=strtobool, default=False, help="load from share partition")
    parser.add_argument("-sj", "--slurm_job", type=int, default=0, help="slurm job")
    
    args = parser.parse_args()

    #### some initialization
    device = 'cpu'
    if args.use_cuda == True and torch.cuda.is_available() == True:
        device = 'cuda'
        torch.cuda.empty_cache()
        torch.set_default_device(device)

    from_share = ""
    if args.from_share == True:
        from_dir = os.path.join(UM.share_path, 'syntheory_plus')

