import torch.nn as nn
import torch
import torch.nn.functional as F

# design model
class SSnet(nn.Module):
    def __init__(self):
        super(SSnet, self).__init__()

        # the input shape should be: (Channel=8, Time=7, Freq=337)
        self.conv1 = nn.Sequential(
            nn.Conv2d(8, 32, (1, 7), (1, 3), (0, 0)), # (32, 7, 110)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, (1, 5), (1, 2), (0, 0)), # (32, 7, 52)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            # nn.BatchNorm2d(128, affine=False), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128)
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128)
        )

        self.conv_6 = nn.Sequential(
            nn.Conv2d(128, 360, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(360), nn.ReLU(inplace=True)
        )

        self.conv_7 = nn.Sequential(
            nn.Conv2d(54, 500, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(500), nn.ReLU(inplace=True)
        )

        self.conv_8 = nn.Sequential(
            nn.Conv2d(500, 1, kernel_size=(7, 5), stride=(1, 1), padding=(0, 2)),
            nn.BatchNorm2d(1), nn.ReLU(inplace=True)
        )

        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, a):
        # a.shape: [B, 8, 7, 337]
        a = self.conv1(a) # [B, 128, 7, 54]
        a_azi = self.relu(a+self.conv_1(a)) # [B, 128, 7, 54]
        a_azi = self.relu(a_azi+self.conv_2(a_azi)) # [B, 128, 7, 54]
        a_azi = self.relu(a_azi+self.conv_3(a_azi)) # [B, 128, 7, 54]
        a_azi = self.relu(a_azi+self.conv_4(a_azi)) # [B, 128, 7, 54]
        a_azi = self.relu(a_azi+self.conv_5(a_azi)) # [B, 128, 7, 54]
        a_azi0 = self.conv_6(a_azi) # [B, 360, 7, 54]
        a_azi = a_azi0.permute(0, 3, 2, 1) # [B, 54, 7, 360]
        a_azi = self.conv_7(a_azi) # [B, 500, 7, 360]
        a_azi = self.conv_8(a_azi) # [B, 1, 1, 360]
        a_azi1 = a_azi.view(a_azi.size(0), -1) # [B, 360]

        return a_azi0, a_azi1


# design model
class ASDnet(nn.Module):
    def __init__(self):
        super(ASDnet, self).__init__()

        # before fed into the conv, the SPS value of the sample are normalized
        self.conv_9 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(16, 16, kernel_size=(1, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.conv_10 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.conv_11 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(64, 64, kernel_size=(1, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.conv_12 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(128, 128, kernel_size=(1, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.conv_13 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(1, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(256, 256, kernel_size=(1, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.conv_14 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(512, 512, kernel_size=(1, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.pool = nn.MaxPool2d(2, stride=2)

        self.flat = nn.Flatten()

        # binary classifier is used to tell there is sound source or not.
        self.son = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Dropout(p=0.2),
            nn.Linear(256, 32),
            nn.Dropout(p=0.2),
            nn.Linear(32, 1),
            nn.Dropout(p=0.2),
        )

        self.doa = nn.Sequential(
            nn.Linear(1024, 360),
            nn.Dropout(p=0.2),
            nn.Sigmoid()
        )

    def forward(self, a_azi):
        # a_azi.shape: [B, 1, 1, 360]
        a_sps = self.conv_9(a_azi) # [B, 16, 2, 180]
        # print(f'after conv_9 a_sps.shape {a_sps.shape}')
        a_sps = self.conv_10(a_sps) # [B, 32, 3, 90]
        # print(f'after conv_10 a_sps.shape {a_sps.shape}')
        a_sps = self.conv_11(a_sps) # [B, 64, 3, 45]
        # print(f'after conv_11 a_sps.shape {a_sps.shape}')
        a_sps = self.conv_12(a_sps) # [B, 128, 3, 22]
        # print(f'after conv_12 a_sps.shape {a_sps.shape}')
        a_sps = self.conv_13(a_sps) # [B, 256, 3, 11]
        # print(f'after conv_13 a_sps.shape {a_sps.shape}')
        a_sps = self.conv_14(a_sps) # [B, 512, 3, 5]
        # print(f'after conv_14 a_sps.shape {a_sps.shape}')
        a_sps = self.pool(a_sps) # [B, 512, 1, 2]
        # print(f'after pool a_sps.shape {a_sps.shape}')
        a_flat = self.flat(a_sps) # [B, 1024]
        # print(f'after flat a_flat.shape {a_flat.shape}')
        a_son = self.son(a_flat) # [B, 1]
        # print(f'a_son.shape {a_son.shape}')
        a_doa = self.doa(a_flat) # [B, 360]
        # print(f'a_doa.shape {a_doa.shape}')

        return a_son, a_doa



# design model
class SSLnet(nn.Module):
    def __init__(self, ssnet, asdnet):
        super(SSLnet, self).__init__()
        self.ssnet = ssnet
        self.asdnet = asdnet

    def forward(self, x):
        a_1, pred_sps = self.ssnet(x) # pred_sps.shape [B, 360]
        pred_sps1 = pred_sps.unsqueeze(1).unsqueeze(1)
        pred_son, pred_doa = self.asdnet(pred_sps1) 
        return a_1, pred_sps, pred_son, pred_doa