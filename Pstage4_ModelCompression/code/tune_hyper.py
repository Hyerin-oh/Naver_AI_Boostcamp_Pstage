import argparse
import copy
import optuna
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataloader import create_dataloader
from src.model import Model
from src.utils.torch_utils import model_info
from src.utils.common import read_yaml
from src.utils.macs import calc_macs
from src.trainer import TorchTrainer
from typing import Any, Dict, List, Tuple, Union
from train_hyper import train

import wandb

MODEL_CONFIG = read_yaml(cfg="/opt/ml/p4-opt-6-zeroki/code/exp/hyper_heejun_06_10/model.yaml")
DATA_CONFIG = read_yaml(cfg="/opt/ml/p4-opt-6-zeroki/code/exp/hyper_heejun_06_10/data.yaml")

def search_hyperparam(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """Search hyperparam from user-specified search space."""
    epochs = 50     #trial.suggest_int("epochs", low=50, high=200, step=50)
    img_size = 64  #trial.suggest_categorical("img_size", [96, 112, 168, 224])
    n_select = 0    #trial.suggest_int("n_select", low=0, high=6, step=2)
    batch_size = 64
    init_lr = trial.suggest_categorical('init_lr', [3e-5  , 5e-5 , 7e-5 , 1e-4,  3e-4 ,  5e-4 , 6e-4 , 7e-4])
    gap = trial.suggest_categorical("gap" , [0.1 , 0.01]) 
    opt_name = 'ADAMW'
    weight_decay = 1e-3
    return {
        "EPOCHS": epochs,
        "IMG_SIZE": img_size,
        "n_select": n_select,
        "BATCH_SIZE": batch_size,
        "INIT_LR": init_lr,
        "OPT_NAME": opt_name,
        "GAP" : gap, 
        'WEIGHT_DECAY' : weight_decay     
    }

def objective(trial: optuna.trial.Trial, device) -> Tuple[float, int, float]:
    """Optuna objective.

    Args:
        trial
    Returns:
        float: score1(e.g. accuracy)
        int: score2(e.g. params)
    """
    model_config = copy.deepcopy(MODEL_CONFIG)
    data_config = copy.deepcopy(DATA_CONFIG)

    # hyperparams: EPOCHS, IMG_SIZE, n_select, BATCH_SIZE
    hyperparams = search_hyperparam(trial)

    model_config["input_size"] = [hyperparams["IMG_SIZE"], hyperparams["IMG_SIZE"]]
    
    data_config["AUG_TRAIN_PARAMS"]["n_select"] = hyperparams["n_select"]
    data_config["BATCH_SIZE"] = hyperparams["BATCH_SIZE"]
    data_config["EPOCHS"] = hyperparams["EPOCHS"]
    data_config["IMG_SIZE"] = hyperparams["IMG_SIZE"]
    data_config["INIT_LR"] = hyperparams["INIT_LR"]
    data_config["OPT_NAME"] = hyperparams["OPT_NAME"]
    data_config["WEIGHT_DECAY"] = hyperparams["WEIGHT_DECAY"]
    data_config["GAP"] = hyperparams["GAP"]

    print(data_config)
    log_dir = os.path.join("final_hyp_exp", datetime.now().strftime(f"hyp_No_Trial_{trial.number}_%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    run = wandb.init(entity="zeroki",
                    project='seonghoon',
                    group="mathKINGseonghoon",
                    name=f'Trial_{trial.number}',
                    config=hyperparams,
                    reinit=True
                    )
     
    # model_config, data_config
    _, test_f1, _ = train(
        model_config=model_config,
        data_config=data_config,
        log_dir=log_dir,
        fp16=data_config["FP16"],
        device=device,
    )

    model_instance = Model(model_config, verbose=False)
    #macs = calc_macs(model_instance.model, (3, data_config["IMG_SIZE"], data_config["IMG_SIZE"]))
    wandb.log({'test_f1':test_f1})
    return test_f1

def tune(gpu_id: int, storage: Union[str, None] = None, study_name: str = "mathKINGseonghoon"):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    elif 0 <= gpu_id < torch.cuda.device_count():
        device = torch.device(f"cuda:{gpu_id}")
    sampler = optuna.samplers.TPESampler(n_startup_trials=20)
    if storage is not None:
        rdb_storage = optuna.storages.RDBStorage(url=storage)
    else:
        rdb_storage = None

    study = optuna.create_study(
        directions=["maximize"],
        storage="postgresql://optuna:00000000@118.67.134.200:6011/hr_optuna",
        study_name=study_name,
        sampler=sampler,    
        load_if_exists=True
    )
    study.optimize(lambda trial: objective(trial, device), n_trials=200)

    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trials:")
    best_trials = study.best_trials

    ## trials that satisfies Pareto Fronts
    for tr in best_trials:
        print(f"  value1:{tr.values[0]}, value2:{tr.values[1]}")
        for key, value in tr.params.items():
            print(f"    {key}:{value}")

    best_trial = get_best_trial_with_condition(study)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna tuner.")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
    parser.add_argument("--storage", default="", type=str, help="RDB Storage URL for optuna.")
    parser.add_argument("--study-name", default="mathKINGseonghoon", type=str, help="Optuna study name.")
    args = parser.parse_args()
    tune(args.gpu, storage=None if args.storage == "" else args.storage, study_name=args.study_name)