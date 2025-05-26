from random import random
from dataloader import *
from config import get_config
from solver import Solver
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import json
import sys


def collate_fn(batch):
    """
    Collate functions assume batch = [Dataset[i] for i in index_set]
    """
    try:
        datas = torch.tensor([sample[0] for sample in batch], dtype=torch.float32)
        labels = torch.tensor([sample[1] for sample in batch], dtype=torch.float32)
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        for i, sample in enumerate(batch):
            print(f"Sample {i}: {sample}")
        raise e
    return datas, labels

def dataset_to_dataloader_v2(datasets,shuffle=False):
    if len(datasets)!=0:
        combined_dataset = ConcatDataset(datasets)
    else:
        combined_dataset = datasets
    dataloader = DataLoader(
        dataset=combined_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    return dataloader

if __name__ == "__main__":

    ID = [
        "EC108.npz",
        "EC113.npz",
        "EC122.npz",
        "EC125.npz",
        "EC129.npz",
        "EC133.npz",
        "EC137.npz",
        "EC139.npz",
        "EC142.npz",
        "EC150.npz",
        "EC152.npz",
        "EC153.npz",
        "EC162.npz",
        "EC175.npz",
        "EC81.npz",
        "EC82.npz",
        "EC84.npz",
        "EC87.npz",
        "EC91.npz",
        "EC92.npz",
        "EC96.npz",
        "EC99.npz",
    ]

    # Setting random seed
    random_seed = 336
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

    # Setting the config for each stage
    finetune_id_list = list(range(0,22))

    lr = 0.0003
    config = None
    finetune_fold = False

    valid_label=[]
    valid_pred=[]
    test_label=[]
    test_pred=[]

    finetune_before_label=[]
    finetune_before_pred=[]
    finetune_after_label=[]
    finetune_after_pred=[]

    increase_before_label=[]
    increase_before_pred=[]
    increase_after_label=[]
    increase_after_pred=[]

    
    stim_before_label = []
    stim_before_pred = []
    stim_after_label = []
    stim_after_pred = []

    record_id_list = []
    memo = "None"
    result_path = "./Exp"
    for finetune_id in finetune_id_list:
        config = get_config(
            parse=False,
            n_epoch=30,
            learning_rate = lr,
            finetune_id = finetune_id,
            memo=memo,
        )
        print(config)

        (
            train_dataloaders,
            valid_dataloaders,
            test_dataloaders,
            finetune_data_list,
            increase_data_list,
            finetune_cnt,
            stim_data_loaders,
            label_process,
            stim_list
        ) = dataloader_finetune(
            finetune_id=finetune_id,
            config=config,
        )

        solver = Solver(
            config,
            train_dataloaders,
            valid_dataloaders,
            test_dataloaders,
            None,None,None,None,
            stim_data_loaders,
            stim_list,label_process,
            finetune_id=finetune_id,
        )
        solver.build()
        solver.train()

        for finetune_index in range(finetune_cnt):
            finetune_test_dataloaders = dataset_to_dataloader_v2([finetune_data_list[finetune_index]])
            increase_test_dataloaders = dataset_to_dataloader_v2([increase_data_list[finetune_index]])

            finetune_train_dataset = ConcatDataset([finetune_data_list[i] for i in range(finetune_cnt) if i != finetune_index])
            increase_train_dataset = ConcatDataset([increase_data_list[i] for i in range(finetune_cnt) if i != finetune_index])

            finetune_train_dataloaders = dataset_to_dataloader_v2([finetune_train_dataset],shuffle=True)
            increase_train_dataloaders = dataset_to_dataloader_v2([increase_train_dataset],shuffle=True)

            solver.finetune_train_dataloaders = finetune_train_dataloaders
            solver.finetune_test_dataloaders = finetune_test_dataloaders
            solver.increase_train_dataloaders = increase_train_dataloaders
            solver.increase_test_dataloaders = increase_test_dataloaders

            solver.finetune_train(data="finetune")
            solver.finetune_train(data="increase")

            temp_finetune_before_label = solver.finetune_before_label
            temp_finetune_before_pred  = solver.finetune_before_pred
            temp_finetune_after_label = solver.finetune_after_label
            temp_finetune_after_pred  = solver.finetune_after_pred

            temp_increase_before_label = solver.increase_before_label
            temp_increase_before_pred  = solver.increase_before_pred
            temp_increase_after_label = solver.increase_after_label
            temp_increase_after_pred  = solver.increase_after_pred

            solver.save_excel()

            record_id_list.append(finetune_id)
            valid_label.append(solver.valid_label)
            valid_pred.append(solver.valid_pred)
            test_label.append(solver.test_label)
            test_pred.append(solver.test_pred)

            finetune_before_label.append(temp_finetune_before_label)
            finetune_before_pred.append(temp_finetune_before_pred)
            finetune_after_label.append(temp_finetune_after_label)
            finetune_after_pred.append(temp_finetune_after_pred)

            increase_before_label.append(temp_increase_before_label)
            increase_before_pred.append(temp_increase_before_pred)
            increase_after_label.append(temp_increase_after_label)
            increase_after_pred.append(temp_increase_after_pred)

            stim_before_label.append(solver.stim_before_label)
            stim_before_pred.append(solver.stim_before_pred)
            stim_after_label.append(solver.stim_after_label)
            stim_after_pred.append(solver.stim_after_pred)



current_time = config.time
data = {
    "memo": memo,
    "record_id_list": record_id_list, 
    "valid_label": valid_label,        
    "valid_pred": valid_pred,
    "test_label": test_label,
    "test_pred": test_pred,
    "finetune_before_label": finetune_before_label,
    "finetune_before_pred": finetune_before_pred,
    "finetune_after_label": finetune_after_label,
    "finetune_after_pred": finetune_after_pred,
    "increase_before_label": increase_before_label,  
    "increase_before_pred": increase_before_pred,
    "increase_after_label": increase_after_label,  
    "increase_after_pred": increase_after_pred,
    'stim_before_label': stim_before_label,
    'stim_before_pred': stim_before_pred,
    'stim_after_label': stim_after_label,
    'stim_after_pred':  stim_after_pred,
}

np_data = {key: np.array(value, dtype=object) for key, value in data.items()}
np.savez(result_path + str(current_time) + '-output.npz', **np_data)