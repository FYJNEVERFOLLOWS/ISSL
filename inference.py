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
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

import audio_dataloader
import func
import model
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader


device = torch.device('cpu')


def infer_one_epoch(epoch: int, val_data_path, test_data_path, model_save_path, pth_path, asd_pth_path):
    infer_start = time.time()
    ssnet = model.SSnet()
    ssnet.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu')))
    asdnet = model.ASDnet()
    asdnet.load_state_dict(torch.load(asd_pth_path, map_location=torch.device('cpu')))

    # Construct loss function and Optimizer.
    criterion_sps = torch.nn.MSELoss()

    val_data = DataLoader(audio_dataloader.VCTK_Dataset(val_data_path), batch_size=100,
                        shuffle=True, num_workers=0)  # val_data is a tuple: (batch_x, batch_y, batch_z)
    test_data = DataLoader(audio_dataloader.VCTK_Dataset(test_data_path), batch_size=100,
                        shuffle=True, num_workers=0)  # test_data is a tuple: (batch_x, batch_y, batch_z)

    # Validate
    cnt_acc_1 = 0
    cnt_acc_2 = 0
    cnt_acc_3 = 0
    cnt_acc_4 = 0
    sum_err_1 = 0
    sum_err_2 = 0
    sum_err_3 = 0
    sum_err_4 = 0
    total = 0 # total doas
    total_0 = 0 # total # of frames which has no sources
    total_1 = 0 # total # of doas in 1 source condition
    total_2 = 0 # total # of doas in mixture consists of 2 sources
    total_3 = 0
    total_4 = 0
    num_pred_half = 0
    num_pred_dot1 = 0
    num_pred_dot2 = 0
    num_pred_dot3 = 0
    num_pred_dot4 = 0
    num_pred_dot6 = 0
    num_pred_dot7 = 0
    num_pred_dot8 = 0
    num_pred_dot9 = 0
    num_acc_half = 0
    num_acc_dot1 = 0
    num_acc_dot2 = 0
    num_acc_dot3 = 0
    num_acc_dot4 = 0
    num_acc_dot6 = 0
    num_acc_dot7 = 0
    num_acc_dot8 = 0
    num_acc_dot9 = 0
    num_target = 0
    sec_val_list = []
    thr_val_list = []
    cnt_acc_sc = 0
    sum_err_sc = 0
    pred_nos_total = 0
    nos_cnt_acc = 0

    nos_total = 0
    epoch_loss = 0.
    sam_size = 0.

    with torch.no_grad():
        ssnet.eval()

        asdnet.eval()
        z_true = []
        z_pred = []
        for index, (batch_x, batch_y, batch_z) in enumerate(val_data):
            batch_x = batch_x.to(device) # batch_x.shape [B, 8, 7, 337]
            batch_y = batch_y.to(device) # batch_y.shape [B, 360]
            batch_z = batch_z.to(device) # batch_z.shape [B,]

            # batch_z.shape[0] = batch_size
            nos_total += batch_z.size(0)
            z_true.extend(batch_z.tolist())

            # 获得模型预测结果
            a_1, output = ssnet(batch_x) # output.shape [B, 360]
            val_loss = criterion_sps(output, batch_y) # averaged loss on batch_y
            with torch.no_grad():
                epoch_loss += batch_z.size(0) * val_loss.clone().detach().item()
                sam_size += batch_z.size(0)

            pred_son, pred_doa = asdnet(output.unsqueeze(1).unsqueeze(1)) # output.shape [B, 360]



            pred_nos = []
            for batch in range(batch_z.size(0)):
                label_nos_batch = batch_z[batch].item()
                pred_nos_batch = 0
                sps_temp = output[batch]
                pred_son_temp = pred_son[batch]
                # print(f'Val batch {batch} pred_son_temp {pred_son_temp} label_nos_batch {label_nos_batch}', flush=True)

                while torch.sigmoid(pred_son_temp) > 0.5 and pred_nos_batch < 8:
                    pred_nos_batch += 1

                    label_nos_batch -= 1
                    peak, sps_temp = func.pop_peak(sps_temp.detach())
                    # print(f'pred_nos_batch {pred_nos_batch} label_nos_batch {label_nos_batch} loss {loss} peak {peak}', flush=True)
                    pred_son_temp = asdnet(sps_temp.unsqueeze(0).unsqueeze(0).unsqueeze(0))[0][0]

                while label_nos_batch > 0:
                    label_nos_batch -= 1
                    peak, sps_temp = func.pop_peak(sps_temp.detach())
                    # print(f'label_nos_batch {label_nos_batch} loss {loss} peak {peak}', flush=True)
                    pred_son_temp = asdnet(sps_temp.unsqueeze(0).unsqueeze(0).unsqueeze(0))[0][0]

                pred_nos.append(pred_nos_batch)
            # print(f'Val pred_nos {pred_nos} ', flush=True)
            z_pred.extend(pred_nos)

            pred_nos = torch.tensor(pred_nos, dtype=torch.int, device=device)

            nos_cnt_acc += torch.sum(pred_nos == batch_z).item()
            pred_nos_total += torch.sum(pred_nos).item()

            for batch in range(batch_z.size(0)):
                # validate for known number of sources
                num_sources = batch_z[batch].item()
                total += num_sources

                label = torch.where(batch_y[batch] == 1)[0]

                if num_sources == 3:
                    thr_val, pred = func.get_top3_doa(output[batch])
                    thr_val_list.append(thr_val)
                    
                if num_sources == 0:
                    total_0 += 1
                
                elif num_sources == 1:
                    pred = torch.max(output[batch], 0)[1]
                    abs_err = func.angular_distance(pred, label)

                    if abs_err <= 5:
                        cnt_acc_1 += 1
                    sum_err_1 += abs_err
                    total_1 += 1
                elif num_sources == 2:
                    sec_val, pred = func.get_top2_doa(output[batch])
                    sec_val_list.append(sec_val)
                    
                    # pred = torch.tensor(pred_cpu, dtype=torch.int, device=device)

                    error = func.angular_distance(pred.reshape([2, 1]), label.reshape([1, 2]))
                    if error[0, 0]+error[1, 1] <= error[1, 0]+error[0, 1]:
                        abs_err = np.array([error[0, 0], error[1, 1]])
                    else:
                        abs_err = np.array([error[0, 1], error[1, 0]])
                    # print(f'pred {pred} label {label} abs_err {abs_err}')
                    cnt_acc_2 += np.sum(abs_err <= 5)
                    sum_err_2 += abs_err.sum()
                    total_2 += 2

                else:
                    pred = torch.tensor(func.get_topk_doa(output[batch], num_sources), dtype=torch.int, device=device)
                    
                    error = func.angular_distance(pred.reshape([num_sources, 1]), label.reshape([1, num_sources]))
                    row_ind, col_ind = linear_sum_assignment(error.cpu())
                    abs_err = error[row_ind, col_ind].cpu().numpy()

                    if num_sources == 3:
                        cnt_acc_3 += np.sum(abs_err <= 5)
                        sum_err_3 += abs_err.sum()
                        total_3 += 3
                    if num_sources == 4:
                        cnt_acc_4 += np.sum(abs_err <= 5)
                        sum_err_4 += abs_err.sum()
                        total_4 += 4
                # val for unknown number of sources
                # Threshold method
                peaks_half = func.get_topk_doa_threshold(output[batch].cpu().detach(), 0.5)
                peaks_half = torch.tensor(peaks_half, dtype=torch.int, device=device)

                peaks_dot1 = func.get_topk_doa_threshold(output[batch].cpu().detach(), 0.1)
                peaks_dot1 = torch.tensor(peaks_dot1, dtype=torch.int, device=device)

                peaks_dot2 = func.get_topk_doa_threshold(output[batch].cpu().detach(), 0.2)
                peaks_dot2 = torch.tensor(peaks_dot2, dtype=torch.int, device=device)

                peaks_dot3 = func.get_topk_doa_threshold(output[batch].cpu().detach(), 0.3)
                peaks_dot3 = torch.tensor(peaks_dot3, dtype=torch.int, device=device)

                peaks_dot4 = func.get_topk_doa_threshold(output[batch].cpu().detach(), 0.4)
                peaks_dot4 = torch.tensor(peaks_dot4, dtype=torch.int, device=device)

                peaks_dot6 = func.get_topk_doa_threshold(output[batch].cpu().detach(), 0.6)
                peaks_dot6 = torch.tensor(peaks_dot6, dtype=torch.int, device=device)

                peaks_dot7 = func.get_topk_doa_threshold(output[batch].cpu().detach(), 0.7)
                peaks_dot7 = torch.tensor(peaks_dot7, dtype=torch.int, device=device)

                peaks_dot8 = func.get_topk_doa_threshold(output[batch].cpu().detach(), 0.8)
                peaks_dot8 = torch.tensor(peaks_dot8, dtype=torch.int, device=device)

                peaks_dot9 = func.get_topk_doa_threshold(output[batch].cpu().detach(), 0.9)
                peaks_dot9 = torch.tensor(peaks_dot9, dtype=torch.int, device=device)
                for l in label:
                    for i in range(l - 5, l + 6):
                        if i in peaks_half:
                            num_acc_half += 1
                        if i in peaks_dot1:
                            num_acc_dot1 += 1
                        if i in peaks_dot2:
                            num_acc_dot2 += 1
                        if i in peaks_dot3:
                            num_acc_dot3 += 1
                        if i in peaks_dot4:
                            num_acc_dot4 += 1
                        if i in peaks_dot6:
                            num_acc_dot6 += 1
                        if i in peaks_dot7:
                            num_acc_dot7 += 1
                        if i in peaks_dot8:
                            num_acc_dot8 += 1
                        if i in peaks_dot9:
                            num_acc_dot9 += 1
                num_pred_half += len(peaks_half)
                num_pred_dot1 += len(peaks_dot1)
                num_pred_dot2 += len(peaks_dot2)
                num_pred_dot3 += len(peaks_dot3)
                num_pred_dot4 += len(peaks_dot4)
                num_pred_dot6 += len(peaks_dot6)
                num_pred_dot7 += len(peaks_dot7)
                num_pred_dot8 += len(peaks_dot8)
                num_pred_dot9 += len(peaks_dot9)

                num_target += label.size(0)

                # SC method
                pred_num_sources = pred_nos[batch].item()
                if pred_num_sources > 0:
                    pred = torch.tensor(func.get_topk_doa(output[batch].cpu().detach(), pred_num_sources), dtype=torch.int, device=device)

                    error = func.angular_distance(pred.reshape([len(pred), 1]), label.reshape([1, num_sources]))
                    row_ind, col_ind = linear_sum_assignment(error.cpu())
                    abs_err = error[row_ind, col_ind].cpu().numpy()

                    cnt_acc_sc += np.sum(abs_err <= 5)
                    sum_err_sc += abs_err.sum()
        sec_val_arr = torch.tensor(sec_val_list, dtype=torch.float)        
        thr_val_arr = torch.tensor(thr_val_list, dtype=torch.float)        

        threshold_mean = torch.mean(sec_val_arr).item()
        threshold_std = torch.std(sec_val_arr).item()
        threshold_mean_3rd = torch.mean(thr_val_arr).item()
        threshold_std_3rd = torch.std(thr_val_arr).item()
        print(f'threshold_mean {threshold_mean} threshold_std {threshold_std}', flush=True)
        print(f'threshold_mean_3rd {threshold_mean_3rd} threshold_std_3rd {threshold_std_3rd}', flush=True)

        cnt_acc = cnt_acc_1 + cnt_acc_2 + cnt_acc_3 + cnt_acc_4
        sum_err = sum_err_1 + sum_err_2 + sum_err_3 + sum_err_4

        epoch_loss = epoch_loss / sam_size

        recall_half = num_acc_half / num_target
        precision_half = num_acc_half / num_pred_half
        F1_half = 2 * recall_half * precision_half / (recall_half + precision_half)

        recall_dot1 = num_acc_dot1 / num_target
        precision_dot1 = num_acc_dot1 / num_pred_dot1
        F1_dot1 = 2 * recall_dot1 * precision_dot1 / (recall_dot1 + precision_dot1)

        recall_dot2 = num_acc_dot2 / num_target
        precision_dot2 = num_acc_dot2 / num_pred_dot2
        F1_dot2 = 2 * recall_dot2 * precision_dot2 / (recall_dot2 + precision_dot2)

        recall_dot3 = num_acc_dot3 / num_target
        precision_dot3 = num_acc_dot3 / num_pred_dot3
        F1_dot3 = 2 * recall_dot3 * precision_dot3 / (recall_dot3 + precision_dot3)

        recall_dot4 = num_acc_dot4 / num_target
        precision_dot4 = num_acc_dot4 / num_pred_dot4
        F1_dot4 = 2 * recall_dot4 * precision_dot4 / (recall_dot4 + precision_dot4)

        recall_dot6 = num_acc_dot6 / num_target
        precision_dot6 = num_acc_dot6 / num_pred_dot6
        F1_dot6 = 2 * recall_dot6 * precision_dot6 / (recall_dot6 + precision_dot6)

        recall_dot7 = num_acc_dot7 / num_target
        precision_dot7 = num_acc_dot7 / num_pred_dot7
        F1_dot7 = 2 * recall_dot7 * precision_dot7 / (recall_dot7 + precision_dot7)

        recall_dot8 = num_acc_dot8 / num_target
        precision_dot8 = num_acc_dot8 / num_pred_dot8
        F1_dot8 = 2 * recall_dot8 * precision_dot8 / (recall_dot8 + precision_dot8)

        recall_dot9 = num_acc_dot9 / num_target
        precision_dot9 = num_acc_dot9 / num_pred_dot9
        F1_dot9 = 2 * recall_dot9 * precision_dot9 / (recall_dot9 + precision_dot9)

        recall_sc = cnt_acc_sc / num_target
        precision_sc = cnt_acc_sc / pred_nos_total
        F1_sc = 2 * recall_sc * precision_sc / (recall_sc + precision_sc)
        MAE_sc = sum_err_sc / pred_nos_total
    print(f'epoch {epoch + 1} epoch_loss {epoch_loss} sam_size {sam_size}', flush=True)
    print(f'epoch {epoch + 1}\'s validation stage, total_zero, {total_0}, total_1 {total_1} total_2 {total_2} total_3 {total_3} total_4 {total_4} total {total}', flush=True)
    print('========== Validation results for known number of sources ==========', flush=True)
    print('Single-source accuracy on val set: %.2f %% ' % (100.0 * cnt_acc_1 / total_1), flush=True)
    print('Single-source MAE on val set: %.3f ' % (sum_err_1 / total_1), flush=True)
    print('Two-sources accuracy on val set: %.2f %% ' % (100.0 * cnt_acc_2 / total_2), flush=True)
    print('Two-sources MAE on val set: %.3f ' % (sum_err_2 / total_2), flush=True)             
    print('Three-sources accuracy on val set: %.2f %% ' % (100.0 * cnt_acc_3 / total_3), flush=True)
    print('Three-sources MAE on val set: %.3f ' % (sum_err_3 / total_3), flush=True)   
    if total_4 > 0:
        print('Four-sources accuracy on val set: %.2f %% ' % (100.0 * cnt_acc_4 / total_4), flush=True)
        print('Four-sources MAE on val set: %.3f ' % (sum_err_4 / total_4), flush=True)   
    print('Overall accuracy on val set: %.2f %% ' % (100.0 * cnt_acc / (total_1 + total_2 + total_3 + total_4)), flush=True)
    print('Overall MAE on val set: %.3f ' % (sum_err / (total_1 + total_2 + total_3 + total_4)), flush=True)

    print(f'Threshold (0.1) method: recall_dot1 {recall_dot1} precision_dot1 {precision_dot1} F1_dot1 {F1_dot1}', flush=True)
    print(f'Threshold (0.2) method: recall_dot2 {recall_dot2} precision_dot2 {precision_dot2} F1_dot2 {F1_dot2}', flush=True)
    print(f'Threshold (0.3) method: recall_dot3 {recall_dot3} precision_dot3 {precision_dot3} F1_dot3 {F1_dot3}', flush=True)
    print(f'Threshold (0.4) method: recall_dot4 {recall_dot4} precision_dot4 {precision_dot4} F1_dot4 {F1_dot4}', flush=True)
    print(f'Threshold (0.5) method: recall_half {recall_half} precision_half {precision_half} F1_half {F1_half}', flush=True)
    print(f'Threshold (0.6) method: recall_dot6 {recall_dot6} precision_dot6 {precision_dot6} F1_dot6 {F1_dot6}', flush=True)
    print(f'Threshold (0.7) method: recall_dot7 {recall_dot7} precision_dot7 {precision_dot7} F1_dot7 {F1_dot7}', flush=True)
    print(f'Threshold (0.8) method: recall_dot8 {recall_dot8} precision_dot8 {precision_dot8} F1_dot8 {F1_dot8}', flush=True)
    print(f'Threshold (0.9) method: recall_dot9 {recall_dot9} precision_dot9 {precision_dot9} F1_dot9 {F1_dot9}', flush=True)    

    print(f'IDOAE method: recall_sc {recall_sc} precision_sc {precision_sc} F1_sc {F1_sc} MAE_sc {MAE_sc}', flush=True)

    print('===== Validation results on the pred of number of sources =====', flush=True)
    print(f'nos_cnt_acc {nos_cnt_acc} nos_total {nos_total} nos_precison {round(100.0 * nos_cnt_acc / nos_total, 3)}', flush=True)

    torch.cuda.empty_cache()
    # 获取混淆矩阵 and normalize by row
    cm_normalized = confusion_matrix(z_true, z_pred, normalize='true')
    print(f'Val Confusion Matrix:\n {np.around(cm_normalized, decimals=2)}', flush=True)

    # Evaluate
    cnt_acc_1 = 0
    cnt_acc_2 = 0
    cnt_acc_3 = 0
    cnt_acc_4 = 0
    sum_err_1 = 0
    sum_err_2 = 0
    sum_err_3 = 0
    sum_err_4 = 0
    total = 0 # total doas
    total_0 = 0 # total # of frames which has no sources
    total_1 = 0 # total # of doas in 1 source condition
    total_2 = 0 # total # of doas in mixture consists of 2 sources
    total_3 = 0
    total_4 = 0
    num_acc = 0
    num_acc_3rd = 0
    num_acc_half = 0
    num_acc_dot1 = 0
    num_acc_dot2 = 0
    num_acc_dot3 = 0
    num_acc_dot4 = 0
    num_acc_dot6 = 0
    num_acc_dot7 = 0
    num_acc_dot8 = 0
    num_acc_dot9 = 0
    num_acc_dot15 = 0
    num_acc_dot25 = 0
    num_acc_dot35 = 0
    num_acc_dot45 = 0
    num_acc_dot55 = 0
    num_acc_dot65 = 0
    num_acc_dot75 = 0
    num_acc_dot85 = 0
    num_acc_dot95 = 0
    num_pred = 0
    num_pred_3rd = 0
    num_pred_half = 0
    num_pred_dot1 = 0
    num_pred_dot2 = 0
    num_pred_dot3 = 0
    num_pred_dot4 = 0
    num_pred_dot6 = 0
    num_pred_dot7 = 0
    num_pred_dot8 = 0
    num_pred_dot9 = 0
    num_pred_dot15 = 0
    num_pred_dot25 = 0
    num_pred_dot35 = 0
    num_pred_dot45 = 0
    num_pred_dot55 = 0
    num_pred_dot65 = 0
    num_pred_dot75 = 0
    num_pred_dot85 = 0
    num_pred_dot95 = 0
    num_target = 0
    sum_err_th = 0
    sum_err_th_3rd = 0
    sum_err_th_half = 0
    cnt_acc_sc = 0
    sum_err_sc = 0
    nos_cnt_acc = 0
    nos_cnt_acc_th = 0
    nos_cnt_acc_thdot1 = 0
    nos_cnt_acc_thdot15 = 0
    nos_cnt_acc_thdot2 = 0
    nos_cnt_acc_thdot25 = 0
    nos_cnt_acc_thdot3 = 0
    nos_cnt_acc_thdot35 = 0
    nos_cnt_acc_thdot4 = 0
    nos_cnt_acc_thdot45 = 0
    nos_cnt_acc_thdot5 = 0
    nos_cnt_acc_thdot55 = 0
    nos_cnt_acc_thdot6 = 0
    nos_cnt_acc_thdot65 = 0
    nos_cnt_acc_thdot7 = 0
    nos_cnt_acc_thdot75 = 0
    nos_cnt_acc_thdot8 = 0
    nos_cnt_acc_thdot85 = 0
    nos_cnt_acc_thdot9 = 0
    nos_cnt_acc_thdot95 = 0
    nos_total = 0
    pred_nos_total = 0

    with torch.no_grad():
        ssnet.eval()
        asdnet.eval()
        z_true = []
        z_pred = []
        for index, (batch_x, batch_y, batch_z) in enumerate(test_data):
            batch_x = batch_x.to(device) # batch_x.shape [B, 8, 7, 337]
            batch_y = batch_y.to(device) # batch_y.shape [B, 360]
            batch_z = batch_z.to(device) # batch_z.shape [B,]

            # batch_z.shape[0] = batch_size
            nos_total += batch_z.size(0)
            z_true.extend(batch_z.tolist())

            # 获得模型预测结果
            a_1, output = ssnet(batch_x) # output.shape [B, 360]

            pred_son, pred_doa = asdnet(output.unsqueeze(1).unsqueeze(1)) # output.shape [B, 360]

            pred_nos = []
            pred_nos_th = []
            pred_nos_thdot1 = []
            pred_nos_thdot15 = []
            pred_nos_thdot2 = []
            pred_nos_thdot25 = []
            pred_nos_thdot3 = []
            pred_nos_thdot35 = []
            pred_nos_thdot4 = []
            pred_nos_thdot45 = []
            pred_nos_thdot5 = []
            pred_nos_thdot55 = []
            pred_nos_thdot6 = []
            pred_nos_thdot65 = []
            pred_nos_thdot7 = []
            pred_nos_thdot75 = []
            pred_nos_thdot8 = []
            pred_nos_thdot85 = []
            pred_nos_thdot9 = []
            pred_nos_thdot95 = []
            for batch in range(batch_z.size(0)):
                label_nos_batch = batch_z[batch].item()
                pred_nos_batch = 0
                sps_temp = output[batch]
                pred_son_temp = pred_son[batch]
                # print(f'Val batch {batch} pred_son_temp {pred_son_temp} label_nos_batch {label_nos_batch}', flush=True)

                while torch.sigmoid(pred_son_temp) > 0.5 and pred_nos_batch < 8:
                    pred_nos_batch += 1

                    label_nos_batch -= 1
                    peak, sps_temp = func.pop_peak(sps_temp.detach())
                    # print(f'pred_nos_batch {pred_nos_batch} label_nos_batch {label_nos_batch} loss {loss} peak {peak}', flush=True)
                    pred_son_temp = asdnet(sps_temp.unsqueeze(0).unsqueeze(0).unsqueeze(0))[0][0]

                while label_nos_batch > 0:
                    label_nos_batch -= 1
                    peak, sps_temp = func.pop_peak(sps_temp.detach())
                    # print(f'label_nos_batch {label_nos_batch} loss {loss} peak {peak}', flush=True)
                    pred_son_temp = asdnet(sps_temp.unsqueeze(0).unsqueeze(0).unsqueeze(0))[0][0]

                pred_nos.append(pred_nos_batch)
            # print(f'Val pred_nos {pred_nos} ', flush=True)
            z_pred.extend(pred_nos)

            pred_nos = torch.tensor(pred_nos, dtype=torch.int, device=device)
            nos_cnt_acc += torch.sum(pred_nos == batch_z).item()
            pred_nos_total += torch.sum(pred_nos).item()


            for batch in range(batch_z.size(0)):
                # test for known number of sources
                num_sources = batch_z[batch].item()
                total += num_sources
                
                label = torch.where(batch_y[batch] == 1)[0]

                if num_sources == 0:
                    total_0 += 1
                
                elif num_sources == 1:
                    pred = torch.max(output[batch], 0)[1]
                    abs_err = func.angular_distance(pred, label)

                    if abs_err <= 5:
                        cnt_acc_1 += 1
                    sum_err_1 += abs_err
                    total_1 += 1
                elif num_sources == 2:
                    sec_val, pred = func.get_top2_doa(output[batch])
                    # pred = torch.tensor(pred_cpu, dtype=torch.int, device=device)
                    error = func.angular_distance(pred.reshape([2, 1]), label.reshape([1, 2]))
                    if error[0, 0]+error[1, 1] <= error[1, 0]+error[0, 1]:
                        abs_err = np.array([error[0, 0], error[1, 1]])
                    else:
                        abs_err = np.array([error[0, 1], error[1, 0]])
                    # print(f'pred {pred} label {label} abs_err {abs_err}', flush=True)
                    cnt_acc_2 += np.sum(abs_err <= 5)
                    sum_err_2 += abs_err.sum()
                    total_2 += 2

                else:
                    pred = torch.tensor(func.get_topk_doa(output[batch], num_sources), dtype=torch.int, device=device)
                    
                    error = func.angular_distance(pred.reshape([num_sources, 1]), label.reshape([1, num_sources]))
                    row_ind, col_ind = linear_sum_assignment(error.cpu())
                    abs_err = error[row_ind, col_ind].cpu().numpy()

                    if num_sources == 3:
                        cnt_acc_3 += np.sum(abs_err <= 5)
                        sum_err_3 += abs_err.sum()
                        total_3 += num_sources
                    if num_sources == 4:
                        cnt_acc_4 += np.sum(abs_err <= 5)
                        sum_err_4 += abs_err.sum()
                        total_4 += num_sources

                # test for unknown number of sources
                # Threshold method
                peaks = func.get_topk_doa_threshold(output[batch].cpu().detach(), threshold_mean - threshold_std)
                peaks = torch.tensor(peaks, dtype=torch.int, device=device)
                pred_nos_th.append(len(peaks))

                
                peaks_3rd = func.get_topk_doa_threshold(output[batch].cpu().detach(), threshold_mean_3rd - threshold_std_3rd)
                peaks_3rd = torch.tensor(peaks_3rd, dtype=torch.int, device=device)

                peaks_half = func.get_topk_doa_threshold(output[batch].cpu().detach(), 0.5)
                peaks_half = torch.tensor(peaks_half, dtype=torch.int, device=device)
                pred_nos_thdot5.append(len(peaks_half))

                peaks_dot1 = func.get_topk_doa_threshold(output[batch].cpu().detach(), 0.1)
                peaks_dot1 = torch.tensor(peaks_dot1, dtype=torch.int, device=device)
                pred_nos_thdot1.append(len(peaks_dot1))

                peaks_dot2 = func.get_topk_doa_threshold(output[batch].cpu().detach(), 0.2)
                peaks_dot2 = torch.tensor(peaks_dot2, dtype=torch.int, device=device)
                pred_nos_thdot2.append(len(peaks_dot2))

                peaks_dot3 = func.get_topk_doa_threshold(output[batch].cpu().detach(), 0.3)
                peaks_dot3 = torch.tensor(peaks_dot3, dtype=torch.int, device=device)
                pred_nos_thdot3.append(len(peaks_dot3))

                peaks_dot4 = func.get_topk_doa_threshold(output[batch].cpu().detach(), 0.4)
                peaks_dot4 = torch.tensor(peaks_dot4, dtype=torch.int, device=device)
                pred_nos_thdot4.append(len(peaks_dot4))

                peaks_dot6 = func.get_topk_doa_threshold(output[batch].cpu().detach(), 0.6)
                peaks_dot6 = torch.tensor(peaks_dot6, dtype=torch.int, device=device)
                pred_nos_thdot6.append(len(peaks_dot6))

                peaks_dot7 = func.get_topk_doa_threshold(output[batch].cpu().detach(), 0.7)
                peaks_dot7 = torch.tensor(peaks_dot7, dtype=torch.int, device=device)
                pred_nos_thdot7.append(len(peaks_dot7))

                peaks_dot8 = func.get_topk_doa_threshold(output[batch].cpu().detach(), 0.8)
                peaks_dot8 = torch.tensor(peaks_dot8, dtype=torch.int, device=device)
                pred_nos_thdot8.append(len(peaks_dot8))

                peaks_dot9 = func.get_topk_doa_threshold(output[batch].cpu().detach(), 0.9)
                peaks_dot9 = torch.tensor(peaks_dot9, dtype=torch.int, device=device)
                pred_nos_thdot9.append(len(peaks_dot9))

                peaks_dot15 = func.get_topk_doa_threshold(output[batch].cpu().detach(), 0.15)
                peaks_dot15 = torch.tensor(peaks_dot15, dtype=torch.int, device=device)
                pred_nos_thdot15.append(len(peaks_dot15))

                peaks_dot25 = func.get_topk_doa_threshold(output[batch].cpu().detach(), 0.25)
                peaks_dot25 = torch.tensor(peaks_dot25, dtype=torch.int, device=device)
                pred_nos_thdot25.append(len(peaks_dot25))

                peaks_dot35 = func.get_topk_doa_threshold(output[batch].cpu().detach(), 0.35)
                peaks_dot35 = torch.tensor(peaks_dot35, dtype=torch.int, device=device)
                pred_nos_thdot35.append(len(peaks_dot35))

                peaks_dot45 = func.get_topk_doa_threshold(output[batch].cpu().detach(), 0.45)
                peaks_dot45 = torch.tensor(peaks_dot45, dtype=torch.int, device=device)
                pred_nos_thdot45.append(len(peaks_dot45))

                peaks_dot55 = func.get_topk_doa_threshold(output[batch].cpu().detach(), 0.55)
                peaks_dot55 = torch.tensor(peaks_dot55, dtype=torch.int, device=device)
                pred_nos_thdot55.append(len(peaks_dot55))

                peaks_dot65 = func.get_topk_doa_threshold(output[batch].cpu().detach(), 0.65)
                peaks_dot65 = torch.tensor(peaks_dot65, dtype=torch.int, device=device)
                pred_nos_thdot65.append(len(peaks_dot65))

                peaks_dot75 = func.get_topk_doa_threshold(output[batch].cpu().detach(), 0.75)
                peaks_dot75 = torch.tensor(peaks_dot75, dtype=torch.int, device=device)
                pred_nos_thdot75.append(len(peaks_dot75))

                peaks_dot85 = func.get_topk_doa_threshold(output[batch].cpu().detach(), 0.85)
                peaks_dot85 = torch.tensor(peaks_dot85, dtype=torch.int, device=device)
                pred_nos_thdot85.append(len(peaks_dot85))

                peaks_dot95 = func.get_topk_doa_threshold(output[batch].cpu().detach(), 0.95)
                peaks_dot95 = torch.tensor(peaks_dot95, dtype=torch.int, device=device)
                pred_nos_thdot95.append(len(peaks_dot95))


                for l in label:
                    for i in range(l - 5, l + 6):
                        if i in peaks:
                            num_acc += 1
                        if i in peaks_3rd:
                            num_acc_3rd += 1
                        if i in peaks_half:
                            num_acc_half += 1
                        if i in peaks_dot1:
                            num_acc_dot1 += 1
                        if i in peaks_dot2:
                            num_acc_dot2 += 1
                        if i in peaks_dot3:
                            num_acc_dot3 += 1
                        if i in peaks_dot4:
                            num_acc_dot4 += 1
                        if i in peaks_dot6:
                            num_acc_dot6 += 1
                        if i in peaks_dot7:
                            num_acc_dot7 += 1
                        if i in peaks_dot8:
                            num_acc_dot8 += 1
                        if i in peaks_dot9:
                            num_acc_dot9 += 1
                        if i in peaks_dot15:
                            num_acc_dot15 += 1
                        if i in peaks_dot25:
                            num_acc_dot25 += 1
                        if i in peaks_dot35:
                            num_acc_dot35 += 1
                        if i in peaks_dot45:
                            num_acc_dot45 += 1
                        if i in peaks_dot55:
                            num_acc_dot55 += 1
                        if i in peaks_dot65:
                            num_acc_dot65 += 1
                        if i in peaks_dot75:
                            num_acc_dot75 += 1
                        if i in peaks_dot85:
                            num_acc_dot85 += 1
                        if i in peaks_dot95:
                            num_acc_dot95 += 1

                num_pred += len(peaks)
                num_pred_3rd += len(peaks_3rd)
                num_pred_half += len(peaks_half)
                num_pred_dot1 += len(peaks_dot1)
                num_pred_dot2 += len(peaks_dot2)
                num_pred_dot3 += len(peaks_dot3)
                num_pred_dot4 += len(peaks_dot4)
                num_pred_dot6 += len(peaks_dot6)
                num_pred_dot7 += len(peaks_dot7)
                num_pred_dot8 += len(peaks_dot8)
                num_pred_dot9 += len(peaks_dot9)
                num_pred_dot15 += len(peaks_dot15)
                num_pred_dot25 += len(peaks_dot25)
                num_pred_dot35 += len(peaks_dot35)
                num_pred_dot45 += len(peaks_dot45)
                num_pred_dot55 += len(peaks_dot55)
                num_pred_dot65 += len(peaks_dot65)
                num_pred_dot75 += len(peaks_dot75)
                num_pred_dot85 += len(peaks_dot85)
                num_pred_dot95 += len(peaks_dot95)

                num_target += label.size(0)

                error = func.angular_distance(peaks.reshape([len(peaks), 1]), label.reshape([1, num_sources]))
                row_ind, col_ind = linear_sum_assignment(error.cpu())
                abs_err = error[row_ind, col_ind].cpu().numpy()
                sum_err_th += abs_err.sum()

                error_3rd = func.angular_distance(peaks_3rd.reshape([len(peaks_3rd), 1]), label.reshape([1, num_sources]))
                row_ind, col_ind = linear_sum_assignment(error_3rd.cpu())
                abs_err_3rd = error_3rd[row_ind, col_ind].cpu().numpy()
                sum_err_th_3rd += abs_err_3rd.sum()

                error_half = func.angular_distance(peaks_half.reshape([len(peaks_half), 1]), label.reshape([1, num_sources]))
                row_ind, col_ind = linear_sum_assignment(error_half.cpu())
                abs_err_half = error_half[row_ind, col_ind].cpu().numpy()
                sum_err_th_half += abs_err_half.sum()

                # SC method
                pred_num_sources = pred_nos[batch].item()
                if pred_num_sources > 0:
                    pred = torch.tensor(func.get_topk_doa(output[batch].cpu().detach(), pred_num_sources), dtype=torch.int, device=device)

                    error = func.angular_distance(pred.reshape([len(pred), 1]), label.reshape([1, num_sources]))
                    row_ind, col_ind = linear_sum_assignment(error.cpu())
                    abs_err = error[row_ind, col_ind].cpu().numpy()

                    cnt_acc_sc += np.sum(abs_err <= 5)
                    sum_err_sc += abs_err.sum()


            
            pred_nos_th = torch.tensor(pred_nos_th, dtype=torch.int, device=device)
            nos_cnt_acc_th += torch.sum(pred_nos_th == batch_z).item()

            pred_nos_thdot1 = torch.tensor(pred_nos_thdot1, dtype=torch.int, device=device)
            nos_cnt_acc_thdot1 += torch.sum(pred_nos_thdot1 == batch_z).item()

            pred_nos_thdot15 = torch.tensor(pred_nos_thdot15, dtype=torch.int, device=device)
            nos_cnt_acc_thdot15 += torch.sum(pred_nos_thdot15 == batch_z).item()

            pred_nos_thdot2 = torch.tensor(pred_nos_thdot2, dtype=torch.int, device=device)
            nos_cnt_acc_thdot2 += torch.sum(pred_nos_thdot2 == batch_z).item()

            pred_nos_thdot25 = torch.tensor(pred_nos_thdot25, dtype=torch.int, device=device)
            nos_cnt_acc_thdot25 += torch.sum(pred_nos_thdot25 == batch_z).item()

            pred_nos_thdot3 = torch.tensor(pred_nos_thdot3, dtype=torch.int, device=device)
            nos_cnt_acc_thdot3 += torch.sum(pred_nos_thdot3 == batch_z).item()

            pred_nos_thdot35 = torch.tensor(pred_nos_thdot35, dtype=torch.int, device=device)
            nos_cnt_acc_thdot35 += torch.sum(pred_nos_thdot35 == batch_z).item()

            pred_nos_thdot4 = torch.tensor(pred_nos_thdot4, dtype=torch.int, device=device)
            nos_cnt_acc_thdot4 += torch.sum(pred_nos_thdot4 == batch_z).item()

            pred_nos_thdot45 = torch.tensor(pred_nos_thdot45, dtype=torch.int, device=device)
            nos_cnt_acc_thdot45 += torch.sum(pred_nos_thdot45 == batch_z).item()

            pred_nos_thdot5 = torch.tensor(pred_nos_thdot5, dtype=torch.int, device=device)
            nos_cnt_acc_thdot5 += torch.sum(pred_nos_thdot5 == batch_z).item()

            pred_nos_thdot55 = torch.tensor(pred_nos_thdot55, dtype=torch.int, device=device)
            nos_cnt_acc_thdot55 += torch.sum(pred_nos_thdot55 == batch_z).item()

            pred_nos_thdot6 = torch.tensor(pred_nos_thdot6, dtype=torch.int, device=device)
            nos_cnt_acc_thdot6 += torch.sum(pred_nos_thdot6 == batch_z).item()

            pred_nos_thdot65 = torch.tensor(pred_nos_thdot65, dtype=torch.int, device=device)
            nos_cnt_acc_thdot65 += torch.sum(pred_nos_thdot65 == batch_z).item()

            pred_nos_thdot7 = torch.tensor(pred_nos_thdot7, dtype=torch.int, device=device)
            nos_cnt_acc_thdot7 += torch.sum(pred_nos_thdot7 == batch_z).item()

            pred_nos_thdot75 = torch.tensor(pred_nos_thdot75, dtype=torch.int, device=device)
            nos_cnt_acc_thdot75 += torch.sum(pred_nos_thdot75 == batch_z).item()

            pred_nos_thdot8 = torch.tensor(pred_nos_thdot8, dtype=torch.int, device=device)
            nos_cnt_acc_thdot8 += torch.sum(pred_nos_thdot8 == batch_z).item()

            pred_nos_thdot85 = torch.tensor(pred_nos_thdot85, dtype=torch.int, device=device)
            nos_cnt_acc_thdot85 += torch.sum(pred_nos_thdot85 == batch_z).item()
            
            pred_nos_thdot9 = torch.tensor(pred_nos_thdot9, dtype=torch.int, device=device)
            nos_cnt_acc_thdot9 += torch.sum(pred_nos_thdot9 == batch_z).item()
            
            pred_nos_thdot95 = torch.tensor(pred_nos_thdot95, dtype=torch.int, device=device)
            nos_cnt_acc_thdot95 += torch.sum(pred_nos_thdot95 == batch_z).item()


        cnt_acc = cnt_acc_1 + cnt_acc_2 + cnt_acc_3 + cnt_acc_4
        sum_err = sum_err_1 + sum_err_2 + sum_err_3 + sum_err_4
        recall = num_acc / num_target
        precision = num_acc / num_pred
        F1 = 2 * recall * precision / (recall + precision)
        MAE = sum_err_th / num_pred

        recall_3rd = num_acc_3rd / num_target
        precision_3rd = num_acc_3rd / num_pred_3rd
        F1_3rd = 2 * recall_3rd * precision_3rd / (recall_3rd + precision_3rd)
        MAE_3rd = sum_err_th_3rd / num_pred_3rd

        recall_half = num_acc_half / num_target
        precision_half = num_acc_half / num_pred_half
        F1_half = 2 * recall_half * precision_half / (recall_half + precision_half)
        MAE_half = sum_err_th_half / num_pred_half

        recall_dot1 = num_acc_dot1 / num_target
        precision_dot1 = num_acc_dot1 / num_pred_dot1
        F1_dot1 = 2 * recall_dot1 * precision_dot1 / (recall_dot1 + precision_dot1)

        recall_dot2 = num_acc_dot2 / num_target
        precision_dot2 = num_acc_dot2 / num_pred_dot2
        F1_dot2 = 2 * recall_dot2 * precision_dot2 / (recall_dot2 + precision_dot2)

        recall_dot3 = num_acc_dot3 / num_target
        precision_dot3 = num_acc_dot3 / num_pred_dot3
        F1_dot3 = 2 * recall_dot3 * precision_dot3 / (recall_dot3 + precision_dot3)

        recall_dot4 = num_acc_dot4 / num_target
        precision_dot4 = num_acc_dot4 / num_pred_dot4
        F1_dot4 = 2 * recall_dot4 * precision_dot4 / (recall_dot4 + precision_dot4)

        recall_dot6 = num_acc_dot6 / num_target
        precision_dot6 = num_acc_dot6 / num_pred_dot6
        F1_dot6 = 2 * recall_dot6 * precision_dot6 / (recall_dot6 + precision_dot6)

        recall_dot7 = num_acc_dot7 / num_target
        precision_dot7 = num_acc_dot7 / num_pred_dot7
        F1_dot7 = 2 * recall_dot7 * precision_dot7 / (recall_dot7 + precision_dot7)

        recall_dot8 = num_acc_dot8 / num_target
        precision_dot8 = num_acc_dot8 / num_pred_dot8
        F1_dot8 = 2 * recall_dot8 * precision_dot8 / (recall_dot8 + precision_dot8)

        recall_dot9 = num_acc_dot9 / num_target
        precision_dot9 = num_acc_dot9 / num_pred_dot9
        F1_dot9 = 2 * recall_dot9 * precision_dot9 / (recall_dot9 + precision_dot9)

        recall_dot15 = num_acc_dot15 / num_target
        precision_dot15 = num_acc_dot15 / num_pred_dot15
        F1_dot15 = 2 * recall_dot15 * precision_dot15 / (recall_dot15 + precision_dot15)

        recall_dot25 = num_acc_dot25 / num_target
        precision_dot25 = num_acc_dot25 / num_pred_dot25
        F1_dot25 = 2 * recall_dot25 * precision_dot25 / (recall_dot25 + precision_dot25)

        recall_dot35 = num_acc_dot35 / num_target
        precision_dot35 = num_acc_dot35 / num_pred_dot35
        F1_dot35 = 2 * recall_dot35 * precision_dot35 / (recall_dot35 + precision_dot35)

        recall_dot45 = num_acc_dot45 / num_target
        precision_dot45 = num_acc_dot45 / num_pred_dot45
        F1_dot45 = 2 * recall_dot45 * precision_dot45 / (recall_dot45 + precision_dot45)

        recall_dot55 = num_acc_dot55 / num_target
        precision_dot55 = num_acc_dot55 / num_pred_dot55
        F1_dot55 = 2 * recall_dot55 * precision_dot55 / (recall_dot55 + precision_dot55)

        recall_dot65 = num_acc_dot65 / num_target
        precision_dot65 = num_acc_dot65 / num_pred_dot65
        F1_dot65 = 2 * recall_dot65 * precision_dot65 / (recall_dot65 + precision_dot65)

        recall_dot75 = num_acc_dot75 / num_target
        precision_dot75 = num_acc_dot75 / num_pred_dot75
        F1_dot75 = 2 * recall_dot75 * precision_dot75 / (recall_dot75 + precision_dot75)

        recall_dot85 = num_acc_dot85 / num_target
        precision_dot85 = num_acc_dot85 / num_pred_dot85
        F1_dot85 = 2 * recall_dot85 * precision_dot85 / (recall_dot85 + precision_dot85)

        recall_dot95 = num_acc_dot95 / num_target
        precision_dot95 = num_acc_dot95 / num_pred_dot95
        F1_dot95 = 2 * recall_dot95 * precision_dot95 / (recall_dot95 + precision_dot95)

        recall_sc = cnt_acc_sc / num_target
        precision_sc = cnt_acc_sc / pred_nos_total
        F1_sc = 2 * recall_sc * precision_sc / (recall_sc + precision_sc)
        MAE_sc = sum_err_sc / pred_nos_total


    
    print(f'epoch {epoch + 1}\'s test stage, total_zero, {total_0}, total_1 {total_1} total_2 {total_2} total_3 {total_3} total_4 {total_4} total {total}', flush=True)
    print('========== Test results on known number of sources ==========', flush=True)
    print('Single-source accuracy on test set: %.2f %% ' % (100.0 * cnt_acc_1 / total_1), flush=True)
    print('Single-source MAE on test set: %.3f ' % (sum_err_1 / total_1), flush=True)
    print('Two-sources accuracy on test set: %.2f %% ' % (100.0 * cnt_acc_2 / total_2), flush=True)
    print('Two-sources MAE on test set: %.3f ' % (sum_err_2 / total_2), flush=True)             
    print('Three-sources accuracy on test set: %.2f %% ' % (100.0 * cnt_acc_3 / total_3), flush=True)
    print('Three-sources MAE on test set: %.3f ' % (sum_err_3 / total_3), flush=True)  
    if total_4 > 0:
        print('Four-sources accuracy on test set: %.2f %% ' % (100.0 * cnt_acc_4 / total_4), flush=True)
        print('Four-sources MAE on test set: %.3f ' % (sum_err_4 / total_4), flush=True)   
    print('Overall accuracy on test set: %.2f %% ' % (100.0 * cnt_acc / (total_1 + total_2 + total_3 + total_4)), flush=True)
    print('Overall MAE on test set: %.3f ' % (sum_err / (total_1 + total_2 + total_3 + total_4)), flush=True)
    print('===== Test results on unknown number of sources =====', flush=True)
    print(f'Threshold (2nd peak) method: recall {recall} precision {precision} F1 {F1} MAE {MAE}', flush=True)
    print(f'Threshold (3rd peak) method: recall_3rd {recall_3rd} precision_3rd {precision_3rd} F1_3rd {F1_3rd} MAE_3rd {MAE_3rd}', flush=True)

    print(f'Threshold (0.1) method: recall_dot1 {recall_dot1} precision_dot1 {precision_dot1} F1_dot1 {F1_dot1}', flush=True)
    print(f'Threshold (0.15) method: recall_dot15 {recall_dot15} precision_dot15 {precision_dot15} F1_dot15 {F1_dot15}', flush=True)
    print(f'Threshold (0.2) method: recall_dot2 {recall_dot2} precision_dot2 {precision_dot2} F1_dot2 {F1_dot2}', flush=True)
    print(f'Threshold (0.25) method: recall_dot25 {recall_dot25} precision_dot25 {precision_dot25} F1_dot25 {F1_dot25}', flush=True)
    print(f'Threshold (0.3) method: recall_dot3 {recall_dot3} precision_dot3 {precision_dot3} F1_dot3 {F1_dot3}', flush=True)
    print(f'Threshold (0.35) method: recall_dot35 {recall_dot35} precision_dot35 {precision_dot35} F1_dot35 {F1_dot35}', flush=True)
    print(f'Threshold (0.4) method: recall_dot4 {recall_dot4} precision_dot4 {precision_dot4} F1_dot4 {F1_dot4}', flush=True)
    print(f'Threshold (0.45) method: recall_dot45 {recall_dot45} precision_dot45 {precision_dot45} F1_dot45 {F1_dot45}', flush=True)
    print(f'Threshold (0.5) method: recall_half {recall_half} precision_half {precision_half} F1_half {F1_half} MAE_half {MAE_half}', flush=True)
    print(f'Threshold (0.55) method: recall_dot55 {recall_dot55} precision_dot55 {precision_dot55} F1_dot55 {F1_dot55}', flush=True)
    print(f'Threshold (0.6) method: recall_dot6 {recall_dot6} precision_dot6 {precision_dot6} F1_dot6 {F1_dot6}', flush=True)
    print(f'Threshold (0.65) method: recall_dot65 {recall_dot65} precision_dot65 {precision_dot65} F1_dot65 {F1_dot65}', flush=True)
    print(f'Threshold (0.7) method: recall_dot7 {recall_dot7} precision_dot7 {precision_dot7} F1_dot7 {F1_dot7}', flush=True)
    print(f'Threshold (0.75) method: recall_dot75 {recall_dot75} precision_dot75 {precision_dot75} F1_dot75 {F1_dot75}', flush=True)
    print(f'Threshold (0.8) method: recall_dot8 {recall_dot8} precision_dot8 {precision_dot8} F1_dot8 {F1_dot8}', flush=True)
    print(f'Threshold (0.85) method: recall_dot85 {recall_dot85} precision_dot85 {precision_dot85} F1_dot85 {F1_dot85}', flush=True)
    print(f'Threshold (0.9) method: recall_dot9 {recall_dot9} precision_dot9 {precision_dot9} F1_dot9 {F1_dot9}', flush=True)
    print(f'Threshold (0.95) method: recall_dot95 {recall_dot95} precision_dot95 {precision_dot95} F1_dot95 {F1_dot95}', flush=True)


    print(f'IDOAE method: recall_sc {recall_sc} precision_sc {precision_sc} F1_sc {F1_sc} MAE_sc {MAE_sc}', flush=True)

    print('===== Test results on the pred of number of sources =====', flush=True)
    print(f'nos_cnt_acc {nos_cnt_acc} nos_total {nos_total} nos_precison {round(100.0 * nos_cnt_acc / nos_total, 3)}', flush=True)
    print(f'nos_cnt_acc_th {nos_cnt_acc_th} nos_total {nos_total} nos_precison_th {round(100.0 * nos_cnt_acc_th / nos_total, 3)}', flush=True)
    print(f'nos_cnt_acc_thdot1 {nos_cnt_acc_thdot1} nos_total {nos_total} nos_precison_thdot1 {round(100.0 * nos_cnt_acc_thdot1 / nos_total, 3)}', flush=True)
    print(f'nos_cnt_acc_thdot15 {nos_cnt_acc_thdot15} nos_total {nos_total} nos_precison_thdot15 {round(100.0 * nos_cnt_acc_thdot15 / nos_total, 3)}', flush=True)
    print(f'nos_cnt_acc_thdot2 {nos_cnt_acc_thdot2} nos_total {nos_total} nos_precison_thdot2 {round(100.0 * nos_cnt_acc_thdot2 / nos_total, 3)}', flush=True)
    print(f'nos_cnt_acc_thdot25 {nos_cnt_acc_thdot25} nos_total {nos_total} nos_precison_thdot25 {round(100.0 * nos_cnt_acc_thdot25 / nos_total, 3)}', flush=True)
    print(f'nos_cnt_acc_thdot3 {nos_cnt_acc_thdot3} nos_total {nos_total} nos_precison_thdot3 {round(100.0 * nos_cnt_acc_thdot3 / nos_total, 3)}', flush=True)
    print(f'nos_cnt_acc_thdot35 {nos_cnt_acc_thdot35} nos_total {nos_total} nos_precison_thdot35 {round(100.0 * nos_cnt_acc_thdot35 / nos_total, 3)}', flush=True)
    print(f'nos_cnt_acc_thdot4 {nos_cnt_acc_thdot4} nos_total {nos_total} nos_precison_thdot4 {round(100.0 * nos_cnt_acc_thdot4 / nos_total, 3)}', flush=True)
    print(f'nos_cnt_acc_thdot45 {nos_cnt_acc_thdot45} nos_total {nos_total} nos_precison_thdot45 {round(100.0 * nos_cnt_acc_thdot45 / nos_total, 3)}', flush=True)
    print(f'nos_cnt_acc_thdot5 {nos_cnt_acc_thdot5} nos_total {nos_total} nos_precison_thdot5 {round(100.0 * nos_cnt_acc_thdot5 / nos_total, 3)}', flush=True)
    print(f'nos_cnt_acc_thdot55 {nos_cnt_acc_thdot55} nos_total {nos_total} nos_precison_thdot55 {round(100.0 * nos_cnt_acc_thdot55 / nos_total, 3)}', flush=True)
    print(f'nos_cnt_acc_thdot6 {nos_cnt_acc_thdot6} nos_total {nos_total} nos_precison_thdot6 {round(100.0 * nos_cnt_acc_thdot6 / nos_total, 3)}', flush=True)
    print(f'nos_cnt_acc_thdot65 {nos_cnt_acc_thdot65} nos_total {nos_total} nos_precison_thdot65 {round(100.0 * nos_cnt_acc_thdot65 / nos_total, 3)}', flush=True)
    print(f'nos_cnt_acc_thdot7 {nos_cnt_acc_thdot7} nos_total {nos_total} nos_precison_thdot7 {round(100.0 * nos_cnt_acc_thdot7 / nos_total, 3)}', flush=True)
    print(f'nos_cnt_acc_thdot75 {nos_cnt_acc_thdot75} nos_total {nos_total} nos_precison_thdot75 {round(100.0 * nos_cnt_acc_thdot75 / nos_total, 3)}', flush=True)
    print(f'nos_cnt_acc_thdot8 {nos_cnt_acc_thdot8} nos_total {nos_total} nos_precison_thdot8 {round(100.0 * nos_cnt_acc_thdot8 / nos_total, 3)}', flush=True)
    print(f'nos_cnt_acc_thdot85 {nos_cnt_acc_thdot85} nos_total {nos_total} nos_precison_thdot85 {round(100.0 * nos_cnt_acc_thdot85 / nos_total, 3)}', flush=True)
    print(f'nos_cnt_acc_thdot9 {nos_cnt_acc_thdot9} nos_total {nos_total} nos_precison_thdot9 {round(100.0 * nos_cnt_acc_thdot9 / nos_total, 3)}', flush=True)
    print(f'nos_cnt_acc_thdot95 {nos_cnt_acc_thdot95} nos_total {nos_total} nos_precison_thdot95 {round(100.0 * nos_cnt_acc_thdot95 / nos_total, 3)}', flush=True)
    torch.cuda.empty_cache()
    # 获取混淆矩阵 and normalize by row
    cm_normalized = confusion_matrix(z_true, z_pred, normalize='true')
    print(f'Test Confusion Matrix:\n {np.around(cm_normalized, decimals=2)}', flush=True)
    print('Time cost of first stage %.2f'%(time.time()-infer_start), flush=True)
    


if __name__ == '__main__':
    val_data_path = "/Work18/2021/fuyanjie/exp_data/sim_audio_vctk/sim_val_data_frame_level"
    test_data_path = "/Work18/2021/fuyanjie/exp_data/sim_audio_vctk/sim_test_1_data"
    model_save_path = "/Work18/2021/fuyanjie/exp_data/exp_cnn_sc/CNN-SC-A1Wdata"
    pth_path = "/Work18/2021/fuyanjie/exp_data/exp_cnn_sc/CNN-SC-2Wdata/Second_Stage_Epoch59.pth"

    parser = argparse.ArgumentParser(description='inference stage')
    parser.add_argument('--epoch', metavar='EPOCH', type=int,
                        default=-1, help='index of epoch')
    parser.add_argument('--val_data_path', metavar='VAL_DATA_PATH', type=str,
                        default=val_data_path, help='path to the validation data')
    parser.add_argument('--test_data_path', metavar='TEST_DATA_PATH', type=str,
                        default=test_data_path, help='path to the test data')
    parser.add_argument('--model_save_path', metavar='MODEL_SAVE_PATH', type=str,
                        default=model_save_path, help='path to the saved model')       
    parser.add_argument('--pth_path', metavar='PTH_PATH', type=str,
                        default=pth_path, help='path to the saved pth model')                                     
    parser.add_argument('--asd_pth_path', metavar='ASD_PTH_PATH', type=str,
                        default=pth_path, help='path to the saved asd pth model')                    
    args = parser.parse_args()

    # Print arguments
    print('========== Print arguments ==========', flush=True)
    for k, v in vars(args).items():
        print(k,' = ',v, flush=True)
    print('========== Print arguments ==========', flush=True)
    infer_one_epoch(args.epoch, args.val_data_path, args.test_data_path, args.model_save_path, args.pth_path, args.asd_pth_path)
