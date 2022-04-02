import os
import pickle
import random
from pathlib import Path
import numpy as np
np.set_printoptions(threshold=np.inf)
import torch
from torch.utils.data import Dataset, DataLoader

# silent_max_ratio = 0.2

class SS_Dataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.total_x, self.total_z = self._read_file()

    def __getitem__(self, index):
        return torch.tensor(self.total_x[index], dtype=torch.float), torch.tensor(self.total_z[index], dtype=torch.long)

    def __len__(self):
        return len(self.total_z)

    def _read_file(self):
        pkl_file = open(self.data_path, 'rb')
        frames = pickle.load(pkl_file)

        total_x = [] 
        total_z = [] # num of sources
        cnt_source_num_0 = 0
        print("read_file starts", flush=True)
        for frame in frames:
            try:
                origin_data = frame
                z = origin_data["num_sources"]
                if z == 0:
                    cnt_source_num_0 += 1
                    # if cnt_source_num_0 > silent_max_ratio * len(total_z):
                    #     continue
                total_z.append(z)  # [len(frames)]

                x = origin_data["sps"]
                total_x.append(x)  # [len(frames), 8, 7, 337]

                if len(total_z) % 10000 == 0:
                    print("{} frames (segments) have been processed".format(len(total_z)), flush=True)
            except Exception as e:
                print(f'Exception {e}')
        print(f'total_samples {len(total_z)}')

        return total_x, total_z

if __name__ == '__main__':
    train_data_path = "/local01/fuyanjie/sps_val"
    train_data = DataLoader(SS_Dataset(train_data_path), batch_size=64, shuffle=True, num_workers=4) # train_data.shape (batch_x, batch_y)
    print(len(train_data)) # len(train_data) is samples / batch_size
    print(next(iter(train_data))[0].shape, next(iter(train_data))[1].shape)



