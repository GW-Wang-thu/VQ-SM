import os
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torchvision


class DmgDataset(data.Dataset):
    
    def __init__(self, data_folder, opt='SDEG', train=True, device='cuda:0'):
        self.train = train
        self.opt = opt
        if opt == "SDEG":
            self.sequence_length = 15
        else:
            self.sequence_length = 16
        folder = os.path.join(data_folder, 'train' if train else 'valid')
        all_npys = os.listdir(folder)
        self.all_opt_npy = [os.path.join(folder, item) for item in all_npys if item.endswith(self.opt+'.npy')]
        self.device = device
        self.blur = torchvision.transforms.GaussianBlur(kernel_size=(3, 3), sigma=(2.0, 2.0))

    def __len__(self):
        return len(self.all_opt_npy)

    def __getitem__(self, idx):
        seq_npy = torch.from_numpy(np.load(self.all_opt_npy[idx])[16-self.sequence_length:, 2:-3, 2:-3].astype('float32')).to(self.device)#.unsqueeze(0)
        if self.opt == "MTDMG" or self.opt == "MDMG":
            seq_npy = self.blur(seq_npy)
        tmp_kws = self.all_opt_npy[idx].split('\\')[-1].split('_')[:3]
        load = torch.from_numpy(np.array([float(tmp_kws[0][1:])/100, float(tmp_kws[1][1:])/1000, float(tmp_kws[2][1:])/100]).astype('float32')).to(self.device)
        return load, seq_npy


class DmgData(data.Dataset):
    def __init__(self, data_path, opt, batch_size, num_workers, device):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.opt = opt
        self.device = device

    def _dataloader(self, train):
        dataset = DmgDataset(self.data_path, opt=self.opt, train=train, device=self.device)
        dataloader = data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=train,
            drop_last=True
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader(True)

    def val_dataloader(self):
        return self._dataloader(False)

    def test_dataloader(self):
        return self._dataloader(False)
