import torch, torch.utils.data as TUD
import optuna, pickle, numpy as np  

import util.util_main as UM
import util.util_data as UD
import util.util_wandb as UW
import util.util_optuna as UO
import util.util_probing as UP
from models.linearnnprobe import LinearNNProbe
from models.standard_scaler import StandardScaler
from probe_dataset import ProbeDataset

from functools import partial
from distutils.util import strtobool
import os, sys time, argparse, tomllib


# statistics gathering: first-pass (standard_scaler)
def train_standard_scaler(datadict, subsetdict, configdict, layer_idx = 0, device = 'cpu', expr_suffix = 0, log_data=True):
    ret = {}
    torch_gen = torch.Generator(device=device)
    train_ds = subsetdict['train_subset']
    scaler = StandardScaler(with_mean = True, with_std = True, use_64bit = configdict['is_64bit'], dim=configdict['model_dim'], use_constant_feature_mask = configdict['standard_scaler_constant_feature_mask'], device = device)
    scaler.eval() # no learnable weights, set anyways

    train_ds.set_layer_idx(layer_idx)
    
    mean_vecs = None
    var_vecs = None
    for epoch_idx in range(configdict['num_epochs']):
        train_dl = TUD.DataLoader(subsetdict['train_subset'], batch_size = config_dict['batch_size'], shuffle=configdict['dataloader_shuffle'], generator=torch_gen)
        for batch_idx, data in enumerate(train_dl):
            ipt, ground_truth = data
            scaler.partial_fit(ipt)
        if log_data == True:
            mean_vecs = UP.accumulate_vecs(mean_vecs, scaler.get_mean())
            var_vecs = UP.accumulate_vecs(var_vecs, scaler.get_var())
    
    ret['scaler'] = scaler
    ret['mean_vecs'] = mean_vecs
    ret['var_vecs'] = var_vecs
    return ret

            


            


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
    parser.add_argument("-sf", "--suffix", type=int, default=0, help="suffix")
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
        from_dir = os.path.join(UM.SHARE_PATH, 'syntheory_plus')
    datadict = UD.load_data_dict(args.dataset)

    cur_ds = ProbeDataset(datadict, args.model_size, layer_idx=0, from_dir = from_dir, to_torch = True, device = device)
    subsetdict = UP.get_train_test_subsets(cur_ds, datadict, train_folds = UM.TRAIN_FOLDS, valid_folds =UM.VALID_FOLDS, test_folds = UM.TEST_FOLDS, train_pct = UM.TRAIN_PCT, test_subpct = UM.TEST_SUBPCT, seed = args.split_seed)

    # wandb stuff
    UW.login()
    wandb_dict = UW.build_initdict(args, datadict, subsetdict)

    cur_study = None
    if args.expr_type == 'standard_scaler':
        for layer_idx in range(wandb_dict['config']['model_num_layers']):
            wandb_dict['config']['layer_idx'] = layer_idx
            run_name = UP.get_run_name(args, layer_idx, is_short = False) 
            short_name = UP.get_run_name(args, layer_idx, is_short = True) 
            wandb_dict['id'] = run_name
            wandb_dict['name'] = short_name
            cur_run = UW.init(wandb_dict)
            scaler_dict = train_standard_scaler(datadict, subsetdict, wandb_dict['config'], layer_idx = layer_idx, device = device, expr_suffix = args.suffix, log_data=True)
            UP.save_scaler(scaler_dict['scaler'], run_name, is_64bit = wandb_dict['config']['is_64bit'])
            UW.log_scaler_mean_var(cur_run, scaler_dict)
            UW.finish_run(cur_run)


    else:
        # optuna stuff
        cur_study = UO.create_or_load_study(args, seed=UM.seed)

