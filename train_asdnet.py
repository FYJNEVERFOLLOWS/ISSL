import argparse
import os
import time
import numpy as np
import torch.nn as nn
import torch
from torch import optim
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
plt.figure(dpi=600) # 将显示的所有图分辨率调高
matplotlib.rcParams['axes.unicode_minus']=False # 显示符号
from sklearn.metrics import confusion_matrix

import sps_loader
import func
import model
from torch.utils.data import DataLoader


train_data_path = "/Work21/2021/fuyanjie/exp_data/exp_asd/3sources/sps_train/sps_train_2W.pkl"
test_data_path = "/Work21/2021/fuyanjie/exp_data/exp_asd/3sources/sps_test/sps_test.pkl"

model_save_path = "/Work21/2021/fuyanjie/exp_data/exp_asd/ASD-3sources2W-0"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# device = torch.device('cpu')
device = torch.device('cuda:0')

pth_path = "/Work18/2021/fuyanjie/exp_data/exp_sps_sc/CNN-SC-B1Wdata/SPSnet_Epoch32.pth"
print(f"train_data_path:\n{train_data_path}", flush=True)
print(f"test_data_path:\n{test_data_path}", flush=True)


asdnet = model.ASDnet()
asdnet.to(device)
print(f"asdnet's summary:\n{asdnet}", flush=True)


# Construct loss function and Optimizer.
criterion_sps = torch.nn.MSELoss()
# criterion_son = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.6, 1], device=device)) # for unbalanced training set, weights should be the reciprocal of the num of samples
criterion_son = torch.nn.BCEWithLogitsLoss() # for unbalanced training set, weights should be the reciprocal of the num of samples
criterion_doa = torch.nn.CrossEntropyLoss(weight=torch.tensor([100] * 360, device=device))

optimizer_sc = optim.Adam(asdnet.parameters(), lr=0.001)
scheduler_sc = optim.lr_scheduler.StepLR(optimizer_sc, step_size=10, gamma=0.5)

train_data = DataLoader(sps_loader.SS_Dataset(train_data_path), batch_size=100,
                        shuffle=True, num_workers=0)  # train_data is a tuple: (batch_x, batch_z)
test_data = DataLoader(sps_loader.SS_Dataset(test_data_path), batch_size=100,
                    shuffle=True, num_workers=0)  # test_data is a tuple: (batch_x, batch_z)

##### args #####
start_plot = 78 # start plotting after epoch XX

def main():
    for epoch in range(50):
        # Train
        running_loss = 0.0

        # training cycle forward, backward, update
        _iter = 0
        epoch_loss = 0.
        sam_size = 0.

        asdnet.train()
        for (batch_x, batch_z) in train_data:
            # 获得一个批次的数据和标签(inputs, labels)
            batch_x = batch_x.to(device) # batch_x.shape [B, 360]
            batch_z = batch_z.to(device) # batch_z.shape [B,]

            inputs = batch_x.unsqueeze(1).unsqueeze(1)
            pred_son, pred_doa = asdnet(inputs) 

            labels = torch.where(batch_z > 0, 1, 0).float()
            # in-place version: 
            # a[a > threshold] = 1
            # a[a <= threshold] = 0
            # print(f'labels {labels} ', flush=True)

            # Compute and print loss    
            loss = criterion_son(pred_son, labels.unsqueeze(1)) # averaged loss on labels

            running_loss += loss.item()

            if _iter % 1000 == 0:
                now_loss = running_loss / 1000
                print('[%d, %5d] loss: %.5f' % (epoch + 1, _iter + 1, now_loss), flush=True)
                running_loss = 0.0
            with torch.no_grad():
                epoch_loss += loss.clone().detach().item() * batch_z.shape[0]
                sam_size += batch_z.shape[0]

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer_sc.zero_grad()
            loss.backward()
            optimizer_sc.step()

            # 一个iter以一个batch为单位   
            _iter += 1
        
        # scheduler_sc.step()
        # torch.cuda.empty_cache()

        # print the MSE and the sample size
        print(f'epoch {epoch + 1} epoch_loss {epoch_loss / sam_size} sam_size {sam_size}', flush=True)

                        
        # Validate
        total = 0
        total_0 = 0
        total_1 = 0 # total # of frames which have 1 source
        total_2 = 0 # total # of frames which have 2 sources
        total_3 = 0 # total # of frames which have 3 sources
        total_4 = 0 # total # of frames which have 4 sources
        nos_cnt_acc = 0
        nos_total = 0
        min_val_loss = float("inf")
        epoch_loss = 0.
        sam_size = 0.

        with torch.no_grad():
            asdnet.eval()
            z_true = []
            z_pred = []
            for index, (batch_x, batch_z) in enumerate(test_data):
                batch_x = batch_x.to(device) # batch_x.shape [B, 360]
                batch_z = batch_z.to(device) # batch_z.shape [B,]

                # batch_z.shape[0] = batch_size
                nos_total += batch_z.size(0)
                z_true.extend(batch_z.tolist())

                # 获得模型预测结果
                inputs = batch_x.unsqueeze(1).unsqueeze(1)
                pred_son, pred_doa = asdnet(inputs) 
                pred_nos = []
                for batch in range(batch_z.size(0)):
                    label_nos_batch = batch_z[batch].item()
                    pred_nos_batch = 0
                    sps_temp = batch_x[batch]
                    pred_son_temp = pred_son[batch]
                    # print(f'Val batch {batch} pred_son_temp {pred_son_temp} label_nos_batch {label_nos_batch}', flush=True)

                    while torch.sigmoid(pred_son_temp) > 0.5 and pred_nos_batch < 8:
                        pred_nos_batch += 1
                        if label_nos_batch > 0:
                            loss = criterion_son(pred_son_temp, torch.tensor([1.0], device=device)) # averaged loss on pred_son[batch]
                            running_loss += loss.item()
                        else:
                            loss = criterion_son(pred_son_temp, torch.tensor([0.0], device=device)) # averaged loss on pred_son[batch]
                            running_loss += loss.item()
                        label_nos_batch -= 1
                        
                        peak, sps_temp = func.pop_peak(sps_temp.detach())
                        # print(f'pred_nos_batch {pred_nos_batch} label_nos_batch {label_nos_batch} loss {loss} peak {peak}', flush=True)
                        pred_son_temp = asdnet(sps_temp.unsqueeze(0).unsqueeze(0).unsqueeze(0))[0][0]
                        with torch.no_grad():
                            epoch_loss += loss.clone().detach().item()
                            sam_size += 1
                    while label_nos_batch > 0:
                        loss = criterion_son(pred_son_temp, torch.tensor([1.0], device=device)) # averaged loss on pred_son[batch]
                        loss.requires_grad = True
                        label_nos_batch -= 1
                        running_loss += loss.item()

                        peak, sps_temp = func.pop_peak(sps_temp.detach())
                        # print(f'label_nos_batch {label_nos_batch} loss {loss} peak {peak}', flush=True)
                        pred_son_temp = asdnet(sps_temp.unsqueeze(0).unsqueeze(0).unsqueeze(0))[0][0]
                   
                        with torch.no_grad():
                            epoch_loss += loss.clone().detach().item()
                            sam_size += 1
                    pred_nos.append(pred_nos_batch)
                z_pred.extend(pred_nos)

                pred_nos = torch.tensor(pred_nos, dtype=torch.int, device=device)

                if _iter % 1000 == 0:
                    now_loss = running_loss / 1000
                    print('[%d, %5d] loss: %.5f' % (epoch + 1, _iter + 1, now_loss), flush=True)
                    running_loss = 0.0
            

                nos_cnt_acc += torch.sum(pred_nos == batch_z).item()
                for batch in range(batch_z.size(0)):
                    num_sources = batch_z[batch].item()

                    if num_sources == 0:
                        total_0 += 1
                    
                    elif num_sources == 1:
                        total += 1
                        total_1 += 1
                    elif num_sources == 2:
                        total += 1
                        total_2 += 1
                    else:
                        if num_sources == 3:
                            total_3 += 1
                        if num_sources == 4:
                            total_4 += 1
                        total += 1
            # 保存模型
            if epoch >= 5 and min_val_loss > epoch_loss:
                min_val_loss = epoch_loss
                save_path = os.path.join(model_save_path, 'ASDnet_Epoch%d.pth'%(epoch+1))
                torch.save(asdnet.state_dict(), save_path)
                print(f'Save model to {save_path}!', flush=True)
                # 获取混淆矩阵 and normalize by row
                cm_normalized = confusion_matrix(z_true, z_pred, normalize='true')
                print(f'Val Confusion Matrix:\n {np.around(cm_normalized, decimals=2)}', flush=True)
        
        print(f'epoch {epoch + 1}\'s validation stage, total_zero, {total_0}, total_1 {total_1} total_2 {total_2} total_3 {total_3} total_4 {total_4} total {total}', flush=True)
        print('===== Validation results on the pred of number of sources =====', flush=True)
        print(f'nos_cnt_acc {nos_cnt_acc} nos_total {nos_total} nos_precison {round(100.0 * nos_cnt_acc / nos_total, 3)}', flush=True)
        torch.cuda.empty_cache()
        



if __name__ == '__main__':
    main()
