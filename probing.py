import torch, optuna, pickle, numpy as np

import util.util_main as UM
import util.util_data as UD
import util.util_wandb as UW
import util.util_optuna as UO
from models.linearnnprobe import LinearNNProbe
from models.standard_scaler import StandardScaler
from probe_dataset import ProbeDataset

from functools import partial
from distutils.util import strtobool
import os, sys time, argparse, tomllib


# statistics gathering: first-pass (standard_scaler)

if __name__ == "__main__":
    #### arg parsing
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ds", "--dataset", type=str, default="polyrhythms", help="dataset")
    parser.add_argument("-ms", "--model_size", type=str, default="small", help="small/medium/large")
    parser.add_argument("-et", "--expr_type", type=str, default="linearnn_full", help="experiment type")
    parser.add_argument("-wdb", "--use_wandb", type=strtobool, default=True, help="sync to wandb")
    parser.add_argument("-cd", "--use_cuda", type=strtobool, default=True, help="use cuda")
    parser.add_argument("-ev", "--eval", type=strtobool, default=False, help="eval")
    parser.add_argument("-rs", "--restart_study", type=strtobool, default=False, help="force restart of optuna study")
    parser.add_argument("-sh", "--from_share", type=strtobool, default=False, help="load from share partition")
    parser.add_argument("-sj", "--slurm_job", type=int, default=0, help="slurm job")
    parser.add_argument("-tsd", "--torch_seed", type=int, default=UM.SEED, help="torch random seed")
    parser.add_argument("-ssd", "--split_seed", type=int, default=UM.SEED, help="seed for splitting")

    args = parser.parse_args()

    #### some initialization
    device = 'cpu'
    if args.use_cuda == True and torch.cuda.is_available() == True:
        device = 'cuda'
        torch.cuda.empty_cache()
        torch.set_default_device(device)
    torch.manual_seed(args.torch_seed)
    from_dir = ""
    if args.from_share == True:
        from_dir = os.path.join(UM.share_path, 'syntheory_plus')
    datadict = UD.load_data_dict(args.dataset)

    cur_ds = ProbeDataset(datadict, args.model_size, layer_idx=0, from_dir = from_dir, to_torch = True, device = device)
    subsetdict = UP.get_train_test_subsets(cur_ds, datadict, train_folds = UM.TRAIN_FOLDS, valid_folds =UM.VALID_FOLDS, test_folds = UM.TEST_FOLDS, train_pct = UM.TRAIN_PCT, test_subpct = UM.TEST_SUBPCT, seed = args.split_seed)

    # wandb stuff
    UW.login()
    wandb_config = UW.build_config(args, datadict, subsetdict)

    cur_study = None
    if args.expr_type == 'standard_scaler':

    else:
        # optuna stuff
        cur_study = UO.create_or_load_study(args, seed=UM.seed)

