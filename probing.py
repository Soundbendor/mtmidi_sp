import torch, torch.utils.data as TUD
from torch import nn
import optuna, pickle, numpy as np  

import util.util_main as UMN
import util.util_metrics as UME
import util.util_constants as UC
import util.util_data as UD
import util.util_wandb as UW
import util.util_optuna as UO
import util.util_probing as UP
import util.util_rdb as UR

from models.mlpprobe import MLPProbe
from models.cae import ConcreteAutoencoder
from models.standard_scaler import StandardScaler
from probe_dataset import ProbeDataset

from functools import partial
from distutils.util import strtobool
import os, sys, time, argparse, copy


# statistics gathering: first-pass (standard_scaler)
def train_standard_scaler(datadict, subsetdict, configdict, layer_idx = 0, device = 'cpu', expr_suffix = 0, log_data=True):
    ret = {}
    # init rng
    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(configdict['torch_seed'])

    train_ds = subsetdict['train_subset']
    scaler = StandardScaler(with_mean = True, with_std = True, use_64bit = configdict['is_64bit'], dim=configdict['model_dim'], use_constant_feature_mask = configdict['standard_scaler_constant_feature_mask'], device = device)
    scaler.eval() # no learnable weights, set anyways

    train_ds.dataset.set_layer_idx(layer_idx)
    
    mean_vecs_batch = None
    var_vecs_batch = None

    mean_vecs_epoch = None
    var_vecs_epoch = None
    for epoch_idx in range(configdict['num_epochs']):
        train_dl = TUD.DataLoader(subsetdict['train_subset'], batch_size = configdict['batch_size'], shuffle=configdict['dataloader_shuffle'], generator=torch_gen)

        if log_data == True and epoch_idx % 20 == 0:
            mean_vecs_epoch = UP.accumulate_vecs(mean_vecs_epoch, scaler.get_mean())
            var_vecs_epoch = UP.accumulate_vecs(var_vecs_epoch, scaler.get_var())
        for batch_idx, data in enumerate(train_dl):
            if log_data == True:
                mean_vecs_batch = UP.accumulate_vecs(mean_vecs_batch, scaler.get_mean())
                var_vecs_batch = UP.accumulate_vecs(var_vecs_batch, scaler.get_var())

            ipt, ground_truth = data
            scaler.partial_fit(ipt)

    if log_data == True:
            mean_vecs_batch = UP.accumulate_vecs(mean_vecs_batch, scaler.get_mean())
            var_vecs_batch = UP.accumulate_vecs(var_vecs_batch, scaler.get_var())
    if log_data == True:
        mean_vecs_epoch = UP.accumulate_vecs(mean_vecs_epoch, scaler.get_mean())
        var_vecs_epoch = UP.accumulate_vecs(var_vecs_epoch, scaler.get_var())
    
    ret['scaler'] = scaler
    ret['mean_vecs_batch'] = mean_vecs_batch
    ret['var_vecs_batch'] = var_vecs_batch

    ret['mean_vecs_epoch'] = mean_vecs_epoch
    ret['var_vecs_epoch'] = var_vecs_epoch
    return ret


def train_model(model, scaler, generator, opt_fn, loss_fn, train_subset, batch_size=64, shuffle = True, is_classification = True, device='cpu'):
    train_dl = TUD.DataLoader(train_subset, batch_size = batch_size, shuffle=shuffle, generator=generator)
    
    total_loss = 0.
    iters = 0
    for batch_idx, data in enumerate(train_dl):
        
        _ipt, ground_truth = data
        ipt = scaler.transform(_ipt)
        model_pred = model(ipt)

        loss = None
        if is_classification == True:
            loss = loss_fn(model_pred, ground_truth)
        else:
            loss = loss_fn(model_pred.flatten(), ground_truth.flatten())
        
        loss.backward()
        opt_fn.step()
        cur_loss = loss.item()
        total_loss += cur_loss
        iters += 1
    avg_loss = total_loss/float(iters)
    return avg_loss

def valid_test_model(model, scaler, generator, loss_fn, valid_subset, batch_size=64, shuffle = True, is_classification = True, device='cpu'):
    valid_dl = TUD.DataLoader(valid_subset, batch_size = batch_size, shuffle=shuffle, generator=generator)
    
    total_loss = 0.
    iters = 0
    avg_loss = 0.
    # for accumulating ground truths and predictions
    truths = None
    preds = None

    for batch_idx, data in enumerate(valid_dl):
        
        _ipt, ground_truth = data
        ipt = scaler.transform(_ipt)
        model_pred = model(ipt)
       
        # don't need loss for testing
        if loss_fn != None:
            loss = None
            if is_classification == True:
                loss = loss_fn(model_pred, ground_truth)
            else:
                loss = loss_fn(model_pred.flatten(), ground_truth.flatten())
            
            cur_loss = loss.item()
            total_loss += cur_loss
            iters += 1

        truths, preds = UP.accumulate_truths_preds(truths, ground_truth, preds, model_pred, batch_idx, is_classification)

    if loss_fn != None:
        avg_loss = total_loss/float(iters)
    return avg_loss, truths, preds

def _objective(trial, datadict, subsetdict, configdict, wandbdict, device='cpu'):
    #dropout = trial.suggest_float('dropout', 0.25, 0.75, step=0.25)

    layer_idx = trial.suggest_categorical('layer_idx', list(range(configdict['model_num_layers'])))
    
    l2_weight_decay_exp = trial.suggest_int('l2_weight_decay_exp', -4, 0, step= 1)
    l2_weight_decay = 0

    if l2_weight_decay_exp < 0:
        l2_weight_decay = 10.**l2_weight_decay_exp

    dropout = trial.suggest_float('dropout', 0., 0.75, step=0.25)

    run_name = UO.get_run_name(configdict, layer_idx, is_short = False)
    trial_number = trial.number
    
    subsetdict['train_subset'].dataset.set_layer_idx(layer_idx)
    subsetdict['valid_subset'].dataset.set_layer_idx(layer_idx)
     
    # load pre-trained scaler
    scaler = StandardScaler(with_mean = True, with_std = True, use_64bit = configdict['is_64bit'], dim=configdict['model_dim'], use_constant_feature_mask = configdict['standard_scaler_constant_feature_mask'], device = device)
    UP.load_scaler_dict(scaler, run_name, is_64bit = configdict['is_64bit'], device=device)
    scaler.eval()

    # init model
    model = MLPProbe(in_dim =configdict['model_dim'], out_dim = datadict['num_classes'], dropout = dropout, initial_dropout = configdict['probe_initial_dropout'], hidden_dims = configdict['probe_hidden_dims'])
    # init rng
    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(configdict['torch_seed'])
    # init opt/loss
    opt_fn = torch.optim.Adam(model.parameters(), lr=configdict['learning_rate'], weight_decay=l2_weight_decay)
    loss_fn = None
    if datadict['is_classification'] == True:
        if datadict['is_balanced'] == True:
            loss_fn = nn.CrossEntropyLoss(reduction='mean')
        else:
            loss_fn = nn.CrossEntropyLoss(reduction='mean', weight = torch.from_numpy(subsetdict['weights']).to(device=device, dtype=(torch.float32 if configdict['is_64bit'] == False else torch.float64)))
    else:
        loss_fn = nn.MSELoss(reduction='mean')

    # other init
    using_early_stopping =  configdict['early_stopping_check_interval'] > 0
    boredom = 0
        
    best_score = float('-inf')
    ret_score = float('-inf')
    accum_metrics = []
    best_model_dict = None
    actual_training_epochs = None

    # wandbstuff
    cur_run = None
    run_name = None
    short_name = None
    if configdict['use_wandb'] == True:
        param_dict = {'l2_weight_decay_exp': l2_weight_decay_exp, 'dropout': dropout}
        run_name, short_name = UO.get_run_and_short_names(configdict, layer_idx, param_dict) 
        cur_run = UW.init(wandbdict, {'id': run_name, 'name': short_name})
        UW.add_to_summary(cur_run, param_dict)
    # now for the actual train/valid loops
    for epoch_idx in range(configdict['num_epochs']):
        # train/valid
        train_avg_loss = train_model(model, scaler, torch_gen, opt_fn, loss_fn, subsetdict['train_subset'], batch_size=configdict['batch_size'], shuffle = configdict['dataloader_shuffle'], is_classification = datadict['is_classification'], device=device)
        valid_avg_loss, valid_truths, valid_preds = valid_test_model(model, scaler, torch_gen, loss_fn, subsetdict['valid_subset'], batch_size=configdict['batch_size'], shuffle = configdict['dataloader_shuffle'], is_classification = datadict['is_classification'], device=device)
        # get validation metrics
        valid_metrics = UME.get_metrics(valid_truths, valid_preds, datadict, configdict, save_to_csv = False, make_cm = False)
        accum_metrics.append(valid_metrics)
        cur_score = UME.get_optimization_metric(valid_metrics, datadict)

        # early stopping
        if using_early_stopping == True:
            if epoch_idx % configdict['early_stopping_check_interval'] == 0:
                if cur_score > best_score:
                    best_score = cur_score
                    boredom = 0
                    best_model_dict = copy.deepcopy(model.state_dict())
                else:
                    boredom += 1
            if boredom >= configdict['early_stopping_boredom']:
                actual_training_epochs = epoch_idx + 1
                ret_score = best_score
                break
            elif epoch_idx == (configdict['num_epochs'] - 1):
                # end of training, just report what you have
                actual_training_epochs = epoch_idx + 1
                best_model_dict = copy.deepcopy(model.state_dict())
                ret_score = cur_score

    # model saving
    if best_model_dict != None:
        UP.save_model_dict(best_model_dict, configdict, layer_idx, trial_number)
    # bookkeeping
    trial.set_user_attr(key='actual_training_epochs', value=actual_training_epochs)
    # naming
    trial.set_user_attr(key='run_name', value=run_name)
    trial.set_user_attr(key='short_name', value=short_name)

    # wandb stuff
    if configdict['use_wandb'] == True:
        UW.log_accum_metrics(cur_run, accum_metrics)
        UW.add_to_summary(cur_run, {'returned_score': ret_score})
        UW.finish_run(cur_run)
    return ret_score

            


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
    parser.add_argument("-sf", "--suffix", type=int, default=1, help="suffix")
    parser.add_argument("-tsd", "--torch_seed", type=int, default=UC.SEED, help="torch random seed")
    parser.add_argument("-ssd", "--split_seed", type=int, default=UC.SEED, help="seed for splitting")

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
        from_dir = os.path.join(UC.SHARE_PATH, 'mtmidi_sp')
    datadict = UD.load_data_dict(args.dataset)

    cur_ds = ProbeDataset(datadict, args.model_size, layer_idx=0, from_dir = from_dir, to_torch = True, device = device)
    subsetdict = UP.get_train_test_subsets(cur_ds, datadict, train_folds = UC.TRAIN_FOLDS, valid_folds =UC.VALID_FOLDS, test_folds = UC.TEST_FOLDS, train_pct = UC.TRAIN_PCT, test_subpct = UC.TEST_SUBPCT, seed = args.split_seed)

    # wandb stuff
    configdict = UW.build_config(args, datadict, subsetdict)
    wandbdict = UW.build_initdict(args, configdict)
    
    if args.eval == False:
        # TRAINING ==========
        if args.use_wandb == True:
            UW.login()
        if args.expr_type == 'standard_scaler':
            for layer_idx in range(configdict['model_num_layers']):
                wandbdict['config']['layer_idx'] = layer_idx
                run_name = UO.get_run_name(configdict, layer_idx, is_short = False) 
                short_name = UO.get_run_name(configdict, layer_idx, is_short = True) 
                wandbdict['id'] = run_name
                wandbdict['name'] = short_name
                cur_run = UW.init(wandbdict)
                scaler_dict = train_standard_scaler(datadict, subsetdict, configdict, layer_idx = layer_idx, device = device, expr_suffix = args.suffix, log_data=True)
                UP.save_scaler_dict(scaler_dict['scaler'], run_name, is_64bit = configdict['is_64bit'])
                UW.log_scaler_batch_mean_var(cur_run, scaler_dict)
                UP.log_scaler_epoch_mean_var(run_name, scaler_dict)
                UW.finish_run(cur_run)


        else:
            # optuna stuff
            studydict = UO.create_or_load_study(args, seed=UC.SEED)
            UO.record_dict_in_study(studydict, configdict)
            objective = partial(_objective, datadict=datadict, subsetdict=subsetdict, configdict=configdict, wandbdict=wandbdict, device=device)
            callback_arr = []
            studydict['study'].optimize(objective, timeout = None, n_trials = None, n_jobs=1, gc_after_trial = True, callbacks=callback_arr)
    else:
        # EVALUATION ========== 

        # load study and get best params given rdb
        cur_study = UO.create_or_load_study(args, seed=UC.SEED)
        best_param_dict, best_trial_dict, attr_dict = UR.get_best_params(cur_study) 
        layer_idx = best_param_dict['layer_idx']['value']
        dropout = best_param_dict['dropout']['value']
        run_name = UO.get_run_name(configdict, layer_idx, is_short = False)

        # some more init
        # init rng
        torch_gen = torch.Generator(device=device)
        torch_gen.manual_seed(configdict['torch_seed'])

        # init/load models
        scaler = StandardScaler(with_mean = True, with_std = True, use_64bit = configdict['is_64bit'], dim=configdict['model_dim'], use_constant_feature_mask = configdict['standard_scaler_constant_feature_mask'], device = device)
        UP.load_scaler_dict(scaler, run_name, is_64bit = configdict['is_64bit'], device=device)
        scaler.eval()

        model = MLPProbe(in_dim =configdict['model_dim'], out_dim = datadict['num_classes'], dropout = dropout, initial_dropout = configdict['probe_initial_dropout'], hidden_dims = configdict['probe_hidden_dims'])

        UP.load_model_dict(model, configdict, layer_idx, trial_number, device=device)
        subsetdict['test_subset'].dataset.set_layer_idx(layer_idx)
        test_avg_loss, test_truths, test_preds = valid_test_model(model, scaler, torch_gen, None, subsetdict['test_subset'], batch_size=configdict['batch_size'], shuffle = configdict['dataloader_shuffle'], is_classification = datadict['is_classification'], device=device)
        # get test metrics
        test_metrics = UME.get_metrics(test_truths, test_preds, datadict, configdict, save_to_csv = True, make_cm = True)

