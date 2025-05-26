import numpy as np
import torch
import os
from config import get_config
from function import *
from scipy.stats import zscore
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torch.utils.data import DataLoader, ConcatDataset

path = None # input benchark data path
stim_data_path = None # input stimulate data path


def search_file_title(path,title):
    list=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith(title):
                list.append(file)
    return list


def split_list(lst):
    length = len(lst)
    size = length // 3
    remainder = length % 3
    splits = []
    for i in range(3):
        if i < remainder:
            splits.append(lst[i*(size+1):(i+1)*(size+1)])
        else:
            splits.append(lst[i*size+remainder:(i+1)*size+remainder])
    return splits

def dataloader_finetune(finetune_id=0, config=None):
    train_id = [
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

    stim_label_list = [
        [], # 0
        [], # 1
        [], # 2
        [61.8056, 64.5833, 61.1111, 34.0277], # 3
        [79.1667, 82.6388, 75.6944], # 4
        [84.0278, 77.0833, 79.1667, 80.5556], # 5
        [], # 6
        [], # 7
        [16.6667, 11.1111, 10.4167, 11.8056], # 8
        [81.2500, 82.3333, 82.6389], # 9
        [50.6944, 59.0278], # 10
        [], # 11
        [], # 12
        [], # 13
        [72.7, 87.9], # 14
        [38.1944,46.6667,47.9167], # 15
        [66.6667,56.6667,70], # 16
        [63.8889,71.5278,65.9722,67.3611], # 17
        [91.6667,89.5833,91.6667,90.9722], # 18
        [65.9722,70.8333], # 19
        [63.1944,58.3333], # 20
        [83.33, 87.5], # 21
    ]

    train_datasets=[]
    
    finetune_id_datasets =[]
    finetune_id_stim = []

    increase_id_datasets = []

    finetune_train=[]
    finetune_tests=[]
    
    num = len(train_id)

    stim_list = []

    max_channel=75

    finetune_data_list=[]
    increase_data_list=[]
    finetune_cnt = 0

    #按人加载
    train_label = []   
    split1,split2,split3 = [],[],[]
    i_split1,i_split2,i_split3 = [],[],[]
    for k in range(0,num):
        splits = []
        stim_datalist = search_file_title(stim_data_path,train_id[k].split(".")[0])
        stim_label = []
        for file in stim_datalist:
            data = np.load(stim_data_path + "/" + file, allow_pickle=True)
            x = data["data"][:, :4, :]
            y = data["label"]
            stim_label.append(y)
            y = np.repeat(y, x.shape[0])
            y = np.expand_dims(y, axis=-1)
            if k == finetune_id:
                stim_list = stim_datalist
                temp = []
                for i, j in zip(x, y):
                    temp.append([i,j])
                finetune_id_stim.append(temp)
        stim_label = np.array(stim_label).mean()

        data = np.load(path + "/" + train_id[k], allow_pickle=True)
        length = len(data["x"])
        size = length // 3
        remainder = length % 3
        for index in range(len(data["x"])):
            x = data["x"][index][:, :4, :]
            y = np.repeat(data["y"][index], x.shape[0])
            y = np.expand_dims(y, axis=-1)
            if k == finetune_id:
                split1 = []
                i_split1 = []
                for i, j in zip(x, y):
                    split1.append([i, j])
                    i_split1.append([i, stim_label - j])
                finetune_data_list.append(split1)
                increase_data_list.append(i_split1)
                finetune_cnt += 1

            else:
                train_label.append(data["y"][index])
                for i, j in zip(x, y):
                    train_datasets.append([i,j])

    # Data Split
    train_split = train_datasets[:int(len(train_datasets)*0.8)]
    test_split = train_datasets[int(len(train_datasets)*0.8):int(len(train_datasets)*0.9)]
    valid_split = train_datasets[int(len(train_datasets)*0.9):]

    train_label = []
    for i in train_split:
        train_label.append(i[1])
    label_process=LabelProcess(np.array(train_label))

    print(f"train data: {len(train_split)}")
    print(f"valid data: {len(valid_split)}")
    print(f"test data: {len(test_split)}")

    print(f"stim data: {len(finetune_id_stim)}")

    

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

    def dataset_to_dataloader_v1(datasets,shuffle=False):
        dataloaders=[]
        for i in datasets:
            temp = DataLoader(
                dataset=i,
                batch_size=config.batch_size,
                shuffle=shuffle,
                collate_fn=collate_fn,
            )
            dataloaders.append(temp)
        return dataloaders

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
    
    train_dataloaders = dataset_to_dataloader_v2([train_split],shuffle=True)
    valid_dataloaders = dataset_to_dataloader_v2([valid_split])
    test_dataloaders = dataset_to_dataloader_v2([test_split])

    # 改顺序
    # finetune_dataloader_list =[]
    # increase_dataloader_list =[]
    # for i in range(finetune_data_cnt):
    #     finetune_dataloader_list.append(dataset_to_dataloader_v2([finetune_data_list[i]]))
    #     increase_dataloader_list.append(dataset_to_dataloader_v2([increase_data_list[i]]))

    # finetune_train_dataloaders = dataset_to_dataloader_v2([finetune_id_datasets[0]],shuffle=True)
    # finetune_test_dataloaders = dataset_to_dataloader_v2([finetune_id_datasets[1]])
    # finetune_valid_dataloaders = dataset_to_dataloader_v2([finetune_id_datasets[2]])

    # increase_train_dataloaders = dataset_to_dataloader_v2([increase_id_datasets[0]],shuffle=True)
    # increase_valid_dataloaders = dataset_to_dataloader_v2([increase_id_datasets[1]])
    # increase_test_dataloaders = dataset_to_dataloader_v2([increase_id_datasets[2]])
                                                          
    stim_data_loaders = dataset_to_dataloader_v1(finetune_id_stim)

    

    return (
        train_dataloaders,
        valid_dataloaders,
        test_dataloaders,
        finetune_data_list,
        increase_data_list,
        finetune_cnt,
        stim_data_loaders,
        label_process,
        stim_list
    )

if __name__ == "__main__":
    for i in range(0,21):
        config = get_config(
            parse=False,
            finetune_id =i,
        )
        (
        train_dataloaders,
        valid_dataloaders,
        test_dataloaders,
        finetune_train_dataloaders,
        finetune_valid_dataloaders,
        finetune_test_dataloaders,
        increase_train_dataloaders,
        increase_valid_dataloaders,
        increase_test_dataloaders,
        stim_data_loaders,
        label_process,
        stim_list
        )=dataloader_finetune(finetune_id=0,config=config)
        exit(0)

