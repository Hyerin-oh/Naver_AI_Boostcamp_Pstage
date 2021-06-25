"""Baseline train
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""

import argparse
from datetime import datetime
import os
import yaml
from typing import Any, Dict, Tuple, Union
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import tensorly as tl
import json
import copy
import pickle

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.dataloader import create_dataloader
from src.augmentation.policies import simple_augment_test
from src.loss import CustomCriterion
from src.model import Model
from src.trainer import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.macs import calc_macs
from src.utils.torch_utils import check_runtime, model_info, seed_everything
from src.scheduler import CosineAnnealingWarmupRestarts
from src.utils.inference_utils import run_model
from Decompose import *  
from inference import * 

def train(
    args,
    model,
    data_config: Dict[str, Any],
    log_dir: str,
    fp16: bool,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train."""
    # save model_config, data_config
    with open(os.path.join(log_dir, 'data.yml'), 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)


    #for reproductin 
    seed_everything(data_config['SEED'])
    
    model_path = os.path.join(log_dir, "best.pt")
    print(f"Model save path: {model_path}")
    # Create dataloader
    train_dl, val_dl, test_dl = create_dataloader(data_config)

    # Calc macs
    macs = calc_macs(model, (3, data_config["IMG_SIZE"], int(data_config["IMG_SIZE"]*0.723)))
    print(f"macs: {macs}")
    
    # Create optimizer, scheduler, criterion
    optimizer = torch.optim.SGD(
        model.parameters(), lr=data_config["INIT_LR"]
    )
    first_cycle_steps = len(train_dl) * data_config["EPOCHS"]
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer, 
        first_cycle_steps=first_cycle_steps, 
        cycle_mult=1.0, 
        max_lr=data_config["INIT_LR"] , 
        min_lr=data_config["INIT_LR"] , 
        warmup_steps=0, 
        gamma=0.5
    )
    
    criterion = CustomCriterion(
        samples_per_cls=get_label_counts(data_config["DATA_PATH"]+'/train')
        if data_config["DATASET"] == "TACO"
        else None,
        device=device,
        fp16 = data_config["FP16"],
        loss_type= "F1", # softmax, logit_adjustment_loss,F1, Focal, LabelSmoothing
        mix = True # if true : loss = 0.25*crossentropy + loss_type
    )
    
    # Amp loss scaler
    scaler = (
        torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None
    )

    # Create trainer
    trainer = TorchTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        model_path=model_path,
        verbose=1,
    )
    best_acc, best_f1 = trainer.train(
        train_dataloader=train_dl,
        n_epoch=data_config["EPOCHS"],
        val_dataloader=val_dl if val_dl else test_dl,
    )

    # evaluate model with test set
    model.load_state_dict(torch.load(model_path))
    test_loss, test_f1, test_acc = trainer.test(
        model=model, test_dataloader=val_dl if val_dl else test_dl
    )
    return test_loss, test_f1, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--model", default="/opt/ml/p4-opt-6-zeroki/code/exp/safebread/model.yml", type=str, help="origin model config"
    )
    parser.add_argument(
        "--data", default="/opt/ml/p4-opt-6-zeroki/code/exp/safebread/data_modi.yml", type=str, help="data config"
    )
    parser.add_argument(
        "--weight", default="/opt/ml/p4-opt-6-zeroki/code/exp/safebread/best.pt", type=str, help="data config"
    )
    parser.add_argument(
        "--run_name", default="heejun", type=str, help="run name for wandb"
    )
    parser.add_argument(
        "--save_name", default="heejun", type=str, help="save name"
    )
    parser.add_argument(
        "--rank_cfg", default="exp_rank/real_37_27.pkl", type=str, help="rank config"
    )
    parser.add_argument(
        "--freeze", default=0, type=int, help="freeze layer except conv"
    )
    parser.add_argument(
	    "--img_root", default = '/opt/ml/input/data/test', type=str, help="image folder root. e.g) 'data/test'"
    )
    parser.add_argument(
        "--dst", default="heejun.csv", type=str, help="destination path for submit"
    )
    args = parser.parse_args()
    model_config = read_yaml(cfg=args.model)
    data_config = read_yaml(cfg=args.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = os.path.join("final", args.save_name + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    with open(args.rank_cfg , 'rb') as f:
        cfg = pickle.load(f)
    print(cfg)
    
    tl.set_backend('pytorch')
    model_instance = Model(model_config, verbose=True)
    model_instance.model.load_state_dict(torch.load(args.weight, map_location=torch.device('cpu')))

    macs = calc_macs(model_instance.model, (3, data_config["IMG_SIZE"], int(data_config["IMG_SIZE"]*0.723)))
    print(f'**************** macs : {macs} ****************')

    decompose_model =copy.deepcopy(model_instance)
    
    idx_list = []
    idx_layer = 0

    for name, param in decompose_model.model.named_modules():
        layer_num = name.split('.')[0]
        if layer_num.isnumeric() and layer_num not in idx_list:
            idx_layer = layer_num
            idx_list.append(idx_layer)
            rank_name = f'{idx_layer}_layer_rank'
            rank1, rank2 = cfg[rank_name + '1'] , cfg[rank_name + '2']
            # rank1 , rank2 = 0.7 , 0.7 # 고정 상수
        if isinstance(param, nn.Conv2d):
            param.register_buffer('rank', torch.Tensor([rank1, rank2])) # rank in, out

    decompose_model.model = decompose(decompose_model.model)
    decompose_model.model.load_state_dict(torch.load('/opt/ml/p4-opt-6-zeroki/code/final/safebread_2021-06-14_22-42-08/best.pt'))
    decompose_model.model.to(device)

    # freeze without conv
    if args.freeze:
        decompose_model.model.requires_grad_(False)
        for name , module in decompose_model.model.named_modules():
            if isinstance(module, nn.Conv2d):
                module.requires_grad_(True)
            
        for name , param in decompose_model.model.named_parameters():
            print(f' {name} : {param.requires_grad}')   
    
    # for wandb
    wandb.init(project='zeroki', entity='zeroki', name = args.run_name , save_code = True)
    wandb.run.name = args.run_name
    wandb.run.save()
    wandb.config.update(data_config)

    # finetune
    test_loss, test_f1, test_acc = train(
        args,
        decompose_model.model,
        data_config=data_config,
        log_dir=log_dir,
        fp16=data_config["FP16"],
        device=device,
    )

    # inference
    dataloader = get_dataloader(img_root=args.img_root, data_config=args.data)
    weight_path = os.path.join(os.path.join('/opt/ml/p4-opt-6-zeroki/code/' ,log_dir) , 'best.pt')
    decompose_model.model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    inference(decompose_model.model, dataloader, device, args.dst)
    print('Done!')