import os

import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback as WBC

import util_main as UM

# https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.WeightsAndBiasesCallback.html

# wandb_kwargs is the things passed to wandb.init()
# https://docs.wandb.ai/models/ref/python/functions/init

# to login
# https://docs.wandb.ai/models/ref/python/functions/login

entity='soundbendor'

def login():
    _key = ''
    with open('wandbkey', 'r') as f:
        _tmp = f.readlines()
        _key = _tmp[0].strip()
    wandb.login(key = _key)

# call directly for standard_scaler
def init(wdict):
    run = wandb.init(
            entity = wdict['entity'], 
            project = wdict['project'],
            dir = wdict['dir'],
            id = wdict['id'],
            name = wdict['name'],
            config = wdict['config']
            )
    return run

def build_config(parser_args, datadict, subsetdict):
    _config = {k:v for (k,v) in vars(parser_args).items()}
    model_shape = UM.get_postacts_shape(parser_args.model_size)
    _config['num_epochs'] = UM.NUM_EPOCHS
    _config['batch_size'] = UM.BATCH_SIZE
    _config['is_64bit'] = UM.IS_64BIT
    _config['model_dim'] = model_shape[1]
    _config['model_num_layers'] = model_shape[0]
    _config['dataloader_shuffle'] = UM.DATALOADER_SHUFFLE
    _config['standard_scaler_constant_feature_mask'] = UM.STANDARD_SCALER_CONSTANT_FEATURE_MASK
    if parser_args.expr_type != 'standard_scaler':
        _config['early_stopping_check_interval'] = UM.EARLY_STOPPING_CHECK_INTERVAL
        _config['early_stopping_boredom'] = UM.EARLY_STOPPING_BOREDOM
    _config['train_folds'] = subsetdict['train_folds']
    _config['valid_folds'] = subsetdict['valid_folds']
    _config['test_folds'] = subsetdict['test_folds']
    _config['is_balanced'] = datadict['is_balanced']
    _config['use_weights'] = subsetdict['weights'].shape[0] > 0
    return _config


def build_initdict(_config, expr_type):
    _d = {'entity': entity, 'project': f'mtmidi_sp-{expr_type}', 'dir': UM.WANDB_PATH}
    _d['config'] = _config
    return _d

def log_scaler_mean_var(cur_run, scalerdict):
    log_dict = {}
    log_dict['mean'] = wandb.plots.HEATMAP('epoch', 'element', scalerdict['mean_vecs'], show_text = False)
    log_dict['var'] = wandb.plots.HEATMAP('epoch', 'element', scalerdict['var_vecs'], show_text = False)
    cur_run.log(log_dict)

def finish_run(cur_run):
    cur_run.finish()
