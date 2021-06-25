"""Tune Model.

- Author: Junghoon Kim, Jongsun Shin
- Contact: placidus36@gmail.com, shinn1897@makinarocks.ai
"""
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
from train import train

import wandb

MODEL_CONFIG = read_yaml(cfg="configs/model/mobilenetv3.yaml")
DATA_CONFIG = read_yaml(cfg="configs/model/tune_taco.yaml")

def search_hyperparam(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """Search hyperparam from user-specified search space."""
    epochs = 50     #trial.suggest_int("epochs", low=50, high=200, step=50)
    img_size = 96  #trial.suggest_categorical("img_size", [96, 112, 168, 224])
    n_select = 0    #trial.suggest_int("n_select", low=0, high=6, step=2)
    batch_size = 64 #trial.suggest_int("batch_size", low=16, high=64, step=16)
    return {
        "EPOCHS": epochs,
        "IMG_SIZE": img_size,
        "n_select": n_select,
        "BATCH_SIZE": batch_size
    }

def search_model(trial: optuna.trial.Trial) -> List[Any]:
    """Search model structure from user-specified search space."""
    model = []
    # 1, 2,3, 4,5, 6,7, 8,9
    # TODO: remove hard-coded stride
    global_output_channel = 3
    UPPER_STRIDE = 1
    # Module 1
    """
    moduel 1 은 stride = 2 , reapeat = 1 이 국룰
    """
    m1 = trial.suggest_categorical("m1", ["Conv", "DWConv"])
    m1_args = []
    m1_repeat = 1
    m1_out_channel = trial.suggest_int("m1/out_channels", low=16, high=24, step=8)
    m1_stride = 2
    m1_activation = trial.suggest_categorical(
        "m1/activation", ["ReLU", "Hardswish"]
        )
    if m1 == "Conv":
        # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
        m1_args = [m1_out_channel, 3, m1_stride, None, 1, m1_activation]
    elif m1 == "DWConv":
        # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
        m1_args = [m1_out_channel, 3, m1_stride, None, m1_activation]
    
    model.append([m1_repeat, m1, m1_args])
    global_output_channel = m1_out_channel

    # Maxpooling 
    model.append([1, 'MaxPool', [3,2,1]])

    # Module 2
    m2 = trial.suggest_categorical(
        "m2",
        ["InvertedResidualv2",
        "InvertedResidualv3",
        "MBConv",
        "ShuffleNetV2"
        ]
    )
    '''
    stride = 2 & repeat = 1로 고정 -> 초반에 resolution을 줄여주기 위함
    '''
    m2_args = []
    m2_sub_args = []
    m2_stride = 2
    m2_repeat = trial.suggest_int("m2/repeat", 2, 4)

    if m2 == "InvertedResidualv2":
        # m2_c = trial.suggest_int("m2/v2_c", low=16, high=32, step=16)
        m2_c = trial.suggest_int("m2/v2_c", low=global_output_channel + 8, high=40, step=8)
        m2_t = trial.suggest_int("m2/v2_t", low=1, high=3)
        m2_args = [m2_c, m2_t, m2_stride]
        m2_sub_args = [m2_c,m2_t , 1]
    elif m2 == "InvertedResidualv3":
        m2_kernel = 3
        # m2_kernel = trial.suggest_int("m2/kernel_size", low=3, high=5, step=2)
        m2_t = round(trial.suggest_float("m2/v3_t", low=1, high=3, step = 0.2),1)
        m2_c = trial.suggest_int("m2/v3_c", low=global_output_channel + 8, high=40, step=8)
        m2_se = trial.suggest_categorical("m2/v3_se", [0, 1])
        m2_hs = trial.suggest_categorical("m2/v3_hs", [0, 1])
        # k t c SE HS s
        m2_args = [m2_kernel, m2_t, m2_c, m2_se, m2_hs, m2_stride]
        m2_sub_args = [m2_kernel, m2_t, m2_c, m2_se, m2_hs, 1]
    elif m2 == "MBConv":
        m2_t = trial.suggest_int("m2/MB_t", low=1, high=3)
        m2_c = trial.suggest_int("m2/MB_c", low=global_output_channel + 8, high=40, step=8)
        m2_kernel = 3
        # m2_kernel = trial.suggest_int("m2/kernel_size", low=3, high=5, step=2)
        m2_args = [m2_t, m2_c, m2_stride, m2_kernel]
        m2_sub_args = [m2_t, m2_c, 1, m2_kernel]
    elif m2 == "ShuffleNetV2":
        m2_c = global_output_channel * 2
        m2_args = [m2_stride]
        m2_sub_args = [1]
    
    model.append([1, m2, m2_args])      # repeat = 1 , stride = 2 로 고정 
    global_output_channel = m2_c

    # Module2의 따까리
    model.append([m2_repeat, m2, m2_sub_args])  # repeat = n , stride = 1

    # Module 3
    m3 = trial.suggest_categorical(
        "m3",
        ["InvertedResidualv2",
        "InvertedResidualv3",
        "MBConv",
        "ShuffleNetV2"
        ]
        )
    '''
    strde = 1 , repeat = 3 ~5 로 열심히 학습해라
    '''
    m3_args = []
    m3_sub_args = []
    m3_stride = 2
    m3_repeat = trial.suggest_int("m3/repeat", 2, 4)

    if m3 == "InvertedResidualv2":
        m3_c = trial.suggest_int("m3/v2_c", low=global_output_channel + 8, high=96, step=8)
        m3_t = trial.suggest_int("m3/v2_t", low=1, high=3)
        m3_args = [m3_c, m3_t, m3_stride]
        m3_sub_args = [m3_c, m3_t, 1]
    elif m3 == "InvertedResidualv3":
        m3_kernel = 3
        m3_t = round(trial.suggest_float("m3/v3_t", low=1, high=3, step = 0.2),1)
        m3_c = trial.suggest_int("m3/v3_c", low=global_output_channel + 8, high=96, step=8)
        m3_se = trial.suggest_categorical("m3/v3_se", [0, 1])
        m3_hs = trial.suggest_categorical("m3/v3_hs", [0, 1])
        m3_args = [m3_kernel, m3_t, m3_c, m3_se, m3_hs, m3_stride]
        m3_sub_args = [m3_kernel, m3_t, m3_c, m3_se, m3_hs, 1]
    elif m3 == "MBConv":
        m3_t = trial.suggest_int("m3/MB_t", low=1, high=3)
        m3_c = trial.suggest_int("m3/MB_c", low=global_output_channel + 8, high=96, step=8)
        m3_kernel = 3
        # trial.suggest_int("m3/kernel_size", low=3, high=5, step=2)
        m3_args = [m3_t, m3_c, m3_stride, m3_kernel]
        m3_sub_args = [m3_t, m3_c, 1, m3_kernel]
    elif m3 == "ShuffleNetV2":
        m3_c = global_output_channel
        m3_args = [m3_stride]
        m3_sub_args = [1]
    
    model.append([1, m3, m3_args])
    global_output_channel = m3_c
        
    # Module3 따까리 
    model.append([m3_repeat, m3, m3_sub_args])

    # Module 4
    m4 = trial.suggest_categorical(
        "m4",
        ["InvertedResidualv2",
        "InvertedResidualv3",
        "MBConv",
        "ShuffleNetV2",
        ]
        )
    m4_args = []
    m4_sub_args = []
    m4_stride = 2
    m4_repeat = trial.suggest_int("m4/repeat", 2, 4)

    if m4 == "InvertedResidualv2":
        m4_c = trial.suggest_int("m4/v2_c", low=global_output_channel + 16, high=160, step=16)
        m4_t = trial.suggest_int("m4/v2_t", low=2, high=3)
        m4_args = [m4_c, m4_t, m4_stride]
        m4_sub_args = [m4_c, m4_t, 1]
    elif m4 == "InvertedResidualv3":
        m4_kernel = 3
        # trial.suggest_int("m4/kernel_size", low=3, high=5, step=2)
        m4_t = round(trial.suggest_float("m4/v3_t", low=2, high=3, step = 0.2),1)
        m4_c = trial.suggest_int("m4/v3_c", low=global_output_channel + 16, high=160, step=16)
        m4_se = trial.suggest_categorical("m4/v3_se", [0, 1])
        m4_hs = trial.suggest_categorical("m4/v3_hs", [0, 1])
        m4_args = [m4_kernel, m4_t, m4_c, m4_se, m4_hs, m4_stride]
        m4_sub_args = [m4_kernel, m4_t, m4_c, m4_se, m4_hs, 1]
    elif m4 == "MBConv":
        m4_t = trial.suggest_int("m4/MB_t", low=2, high=3)
        m4_c = trial.suggest_int("m4/MB_c", low=global_output_channel+16, high=160, step=16)
        m4_kernel = 3
        # trial.suggest_int("m4/kernel_size", low=3, high=5, step=2)
        m4_args = [m4_t, m4_c, m4_stride, m4_kernel]
        m4_sub_args = [m4_t, m4_c, 1, m4_kernel]
    elif m4 == "ShuffleNetV2":
        m4_args = [m4_stride]
        m4_sub_args = [1]
        m4_c = global_output_channel * 2


    model.append([1, m4, m4_args])
    global_output_channel = m4_c

    # Module 4 따가리 
    model.append([m4_repeat, m4, m4_sub_args])

    # Module 5
    m5 = trial.suggest_categorical(
        "m5",
       ["InvertedResidualv2",
        "InvertedResidualv3",
        "MBConv",
        "ShuffleNetV2",
        ]
        )
    m5_args = []
    m5_stride = 1
    # trial.suggest_int("m5/stride", low=1, high=UPPER_STRIDE)
    m5_repeat = trial.suggest_int("m5/repeat", 2, 4)

    if m5 == "InvertedResidualv2":
        m5_c = trial.suggest_int("m5/v2_c", low=global_output_channel + 16, high=256, step=16)
        m5_t = trial.suggest_int("m5/v2_t", low=2, high=4)
        m5_args = [m5_c, m5_t, m5_stride]
    elif m5 == "InvertedResidualv3":
        m5_kernel = 3
        # trial.suggest_int("m5/kernel_size", low=3, high=5, step=2)
        m5_t = round(trial.suggest_float("m5/v3_t", low=2, high=3, step = 0.2),1)
        m5_c = trial.suggest_int("m5/v3_c", low=global_output_channel + 16, high=256, step=16)
        m5_se = trial.suggest_categorical("m5/v3_se", [0, 1])
        m5_hs = trial.suggest_categorical("m5/v3_hs", [0, 1])
        m5_args = [m5_kernel, m5_t, m5_c, m5_se, m5_hs, m5_stride]
    elif m5 == "MBConv":
        m5_t = trial.suggest_int("m5/MB_t", low=2, high=4)
        m5_c = trial.suggest_int("m5/MB_c", low=global_output_channel + 16, high=256, step=16)
        m5_kernel = 3
        # trial.suggest_int("m5/kernel_size", low=3, high=5, step=2)
        m5_args = [m5_t, m5_c, m5_stride, m5_kernel]
    elif m5 == "ShuffleNetV2":
        # m5_c = trial.suggest_int("m5/shuffle_c", low=16, high=32, step=8)
        m5_args = [m5_stride]
        m5_c = global_output_channel

    model.append([m5_repeat, m5, m5_args])
    global_output_channel = m5_c


    # last layer
    last_dim = global_output_channel * trial.suggest_int("last_dim", low=1, high=4, step = 1) # 배율
    # We can setup fixed structure as well
    model.append([1, "GlobalAvgPool", []])
    model.append([1, "Conv", [last_dim, 1, 1]])
    model.append([1, "FixedConv", [9, 1, 1, None, 1, None]])

    return model

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
    model_config["backbone"] = search_model(trial)

    data_config["AUG_TRAIN_PARAMS"]["n_select"] = hyperparams["n_select"]
    data_config["BATCH_SIZE"] = hyperparams["BATCH_SIZE"]
    data_config["EPOCHS"] = hyperparams["EPOCHS"]
    data_config["IMG_SIZE"] = hyperparams["IMG_SIZE"]
    print(data_config)
    log_dir = os.path.join("exp", datetime.now().strftime(f"No_Trial_{trial.number}_%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    wandb.init(entity="zeroki",
                project='CUCU_CUCHEN',
                group="NAS_1",
                name=f'Trial_{trial.number}',
                config=model_config,
                reinit=True
                )

    model_instance = Model(model_config, verbose=False)
    macs = calc_macs(model_instance.model, (3, data_config["IMG_SIZE"], data_config["IMG_SIZE"]))

    if macs>=20000000.0:     
        print(f' trial: {trial.number}, This model has very large macs:{macs}')
        raise optuna.structs.TrialPruned()

    # model_config, data_config
    _, test_f1, _ = train(
        model_config=model_config,
        data_config=data_config,
        log_dir=log_dir,
        fp16=data_config["FP16"],
        device=device,
    )
    # 3명 : automl , 1명 : automl로 augmentation , 1명 : loss바꾸기 -> Full Train
    wandb.log({'test_f1':test_f1,'macs':macs})
    return test_f1, macs

def tune(gpu_id: int, storage: Union[str, None] = None, study_name: str = "NAS"):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    elif 0 <= gpu_id < torch.cuda.device_count():
        device = torch.device(f"cuda:{gpu_id}")
    sampler = optuna.samplers.MOTPESampler(n_startup_trials=20)
    if storage is not None:
        rdb_storage = optuna.storages.RDBStorage(url=storage)
    else:
        rdb_storage = None

    study = optuna.create_study(
        directions=["maximize", "minimize"],
        storage="postgresql://optuna:00000000@118.67.134.200:6011/hr_optuna",   #  내 storage 번호
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
    parser.add_argument("--study-name", default="NAS", type=str, help="Optuna study name.")
    args = parser.parse_args()
    tune(args.gpu, storage=None if args.storage == "" else args.storage, study_name=args.study_name)
