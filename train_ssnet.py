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

import audio_dataloader
import func
import model
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader


train_data_path = "/Work21/2021/fuyanjie/exp_data/sim_audio_vctk/sim_50rooms_2W_A_data"
val_data_path = "/Work21/2021/fuyanjie/exp_data/sim_audio_vctk/sim_4sources_test_6_data"
test_data_path = "/Work21/2021/fuyanjie/exp_data/sim_audio_vctk/sim_4sources_test_5_data"

model_save_path = "/Work21/2021/fuyanjie/exp_data/exp_sps/SS-4sources-0"
device = torch.device('cuda:0')
print(f"train_data_path:\n{train_data_path}", flush=True)
print(f"val_data_path:\n{val_data_path}", flush=True)
print(f"test_data_path:\n{test_data_path}", flush=True)
print(f"model_save_path:\n{model_save_path}", flush=True)


ssnet = model.SSnet()
ssnet.to(device)
print(f"ssnet's summary:\n{ssnet}", flush=True)

# Construct loss function and Optimizer.
criterion_sps = torch.nn.MSELoss()

optimizer_sps = optim.Adam(ssnet.parameters(), lr=0.001)
scheduler_sps = optim.lr_scheduler.StepLR(optimizer_sps, step_size=5, gamma=0.5)


train_data = DataLoader(audio_dataloader.VCTK_Dataset(train_data_path), batch_size=100,
                        shuffle=True, num_workers=0)  # train_data is a tuple: (batch_x, batch_y, batch_z)
val_data = DataLoader(audio_dataloader.VCTK_Dataset(val_data_path), batch_size=100,
                    shuffle=True, num_workers=0)  # val_data is a tuple: (batch_x, batch_y, batch_z)
test_data = DataLoader(audio_dataloader.VCTK_Dataset(test_data_path), batch_size=100,
                    shuffle=True, num_workers=0)  # test_data is a tuple: (batch_x, batch_y, batch_z)

##### args #####
start_plot = 28 # start plotting after epoch XX

def main():
    #######################Two-stage Training: First Stage ##################################
    train_start = time.time()

    ssnet.train()
    for (batch_x, batch_y, batch_z) in train_data:
        labels = batch_y.unsqueeze(-1).unsqueeze(-1).expand(batch_y.size(0), 360, 7, 54).cuda()
        inputs = Variable(batch_x).cuda()
        a_1, pred_sps = ssnet(inputs)  # pred_sps.shape [B, 360]
        loss = criterion_sps(a_1, labels)
        # Backward
        optimizer_sps.zero_grad()
        loss.backward()
        optimizer_sps.step()

    torch.cuda.empty_cache()
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    save_path = os.path.join(model_save_path, 'First_Stage.pth')
    # torch.save(model.state_dict(), save_path)
    print('Time cost of first stage %.2f'%(time.time()-train_start), flush=True)

    #######################Two-stage Training: Second Stage ##################################
    for epoch in range(30):
        # Train
        running_loss = 0.0

        # training cycle forward, backward, update
        _iter = 0
        epoch_loss = 0.
        sam_size = 0.

        ssnet.train()
        for (batch_x, batch_y, batch_z) in train_data:
            # 获得一个批次的数据和标签(inputs, labels)
            batch_x = batch_x.to(device) # batch_x.shape [B, 8, 7, 337]
            batch_y = batch_y.to(device) # batch_y.shape [B, 360]
            batch_z = batch_z.to(device) # batch_z.shape [B,]

            # Forward pass: Compute predicted y by passing x to the model
            a_1, pred_sps = ssnet(batch_x) # pred_sps.shape [B, 360]

            # Compute and print loss    
            loss = criterion_sps(pred_sps, batch_y) # averaged loss on batch_y

            running_loss += loss.item()
            if _iter % 1000 == 0:
                now_loss = running_loss / 1000
                print('[%d, %5d] loss: %.5f' % (epoch + 1, _iter + 1, now_loss), flush=True)
                running_loss = 0.0
            with torch.no_grad():
                epoch_loss += loss.clone().detach().item() * batch_y.shape[0]
                sam_size += batch_y.shape[0]

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer_sps.zero_grad()
            loss.backward()
            optimizer_sps.step()

            # 一个iter以一个batch为单位   
            _iter += 1
        
        scheduler_sps.step()
        torch.cuda.empty_cache()

        # print the MSE and the sample size
        print(f'epoch {epoch + 1} epoch_loss {epoch_loss / sam_size} sam_size {sam_size}', flush=True)

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
        sec_val_list = []
        min_val_loss = 100000
        epoch_loss = 0.
        sam_size = 0.

        with torch.no_grad():
            ssnet.eval()
            for index, (batch_x, batch_y, batch_z) in enumerate(val_data):
                batch_x = batch_x.to(device) # batch_x.shape [B, 8, 7, 337]
                batch_y = batch_y.to(device) # batch_y.shape [B, 360]
                batch_z = batch_z.to(device) # batch_z.shape [B,]

                # batch_z.shape[0] = batch_size

                # 获得模型预测结果
                a_1, output = ssnet(batch_x) # output.shape [B, 360]

                val_loss = criterion_sps(output, batch_y) # averaged loss on batch_y

                with torch.no_grad():
                    epoch_loss += batch_z.size(0) * val_loss.clone().detach().item()
                    sam_size += batch_z.size(0)

                # Plot SPS
                if epoch > start_plot:
                    line_pred, = plt.plot(output[0].cpu(), c='b', label="pred_sps")
                    line_label, = plt.plot(batch_y[0].cpu(), c='r', label="label_sps")
                    plt.title("Comparison between estimated and ground truth SPS")
                    plt.xlabel("DOA")
                    plt.ylabel("Likelihood")
                    plt.legend(handles=[line_pred, line_label])
                    ax=plt.gca() # ax为两条坐标轴的实例
                    ax.xaxis.set_major_locator(MultipleLocator(30)) # 把x轴的主刻度设置为30的倍数
                    # ax.yaxis.set_major_locator(MultipleLocator(1)) # 把y轴的主刻度设置为1的倍数
                    if not os.path.exists(model_save_path + '/diff_sps'):
                        os.makedirs(model_save_path + '/diff_sps')
                    plt.savefig(model_save_path + f'/diff_sps/val_{epoch + 1}th_epoch_{index}th_batch_{0}_nos_{num_sources}.png', dpi=600)
                    plt.close()
                for batch in range(batch_z.size(0)):
                    # validate for known number of sources
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


            sec_val_arr = torch.tensor(sec_val_list, dtype=torch.float)        

            threshold_mean = torch.mean(sec_val_arr).item()
            threshold_std = torch.std(sec_val_arr).item()
            print(f'threshold_mean {threshold_mean} threshold_std {threshold_std}', flush=True)

            cnt_acc = cnt_acc_1 + cnt_acc_2 + cnt_acc_3 + cnt_acc_4
            sum_err = sum_err_1 + sum_err_2 + sum_err_3 + sum_err_4

            epoch_loss = epoch_loss / sam_size
            # 保存模型
            if epoch >= 10 and min_val_loss > epoch_loss:
                min_val_loss = epoch_loss
                save_path = os.path.join(model_save_path, 'SSnet_Epoch%d.pth'%(epoch+1))
                torch.save(ssnet.state_dict(), save_path)
                print(f'Save model to {save_path}!', flush=True)
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
        torch.cuda.empty_cache()


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
        num_pred = 0
        num_target = 0
        sum_err_th = 0


        with torch.no_grad():
            ssnet.eval()
            for index, (batch_x, batch_y, batch_z) in enumerate(test_data):
                batch_x = batch_x.to(device) # batch_x.shape [B, 8, 7, 337]
                batch_y = batch_y.to(device) # batch_y.shape [B, 360]
                batch_z = batch_z.to(device) # batch_z.shape [B,]

                # batch_z.shape[0] = batch_size


                # 获得模型预测结果
                a_1, output = ssnet(batch_x) # output.shape [B, 360]

                for batch in range(batch_z.size(0)):
                    # test for known number of sources
                    num_sources = batch_z[batch].item()
                    total += num_sources


                    # Plot SPS
                    if epoch > start_plot + 10:
                        line_pred, = plt.plot(output[batch].cpu(), c='b', label="pred_sps")
                        line_label, = plt.plot(batch_y[batch].cpu(), c='r', label="label_sps")
                        plt.title("Comparison between estimated and ground truth SPS")
                        plt.xlabel("DOA")
                        plt.ylabel("Likelihood")
                        plt.legend(handles=[line_pred, line_label])
                        ax=plt.gca() # ax为两条坐标轴的实例
                        ax.xaxis.set_major_locator(MultipleLocator(30)) # 把x轴的主刻度设置为30的倍数
                        # ax.yaxis.set_major_locator(MultipleLocator(1)) # 把y轴的主刻度设置为1的倍数
                        if not os.path.exists(model_save_path + '/diff_sps'):
                            os.makedirs(model_save_path + '/diff_sps')
                        plt.savefig(model_save_path + f'/diff_sps/test_{epoch + 1}th_epoch_{index}th_batch_{batch}_nos_{num_sources}.png', dpi=600)
                        plt.close()
                    
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
                    
                    for l in label:
                        for i in range(l - 5, l + 6):
                            if i in peaks:
                                num_acc += 1
                    num_pred += len(peaks)
                    num_target += label.size(0)

                    error = func.angular_distance(peaks.reshape([len(peaks), 1]), label.reshape([1, num_sources]))
                    row_ind, col_ind = linear_sum_assignment(error.cpu())
                    abs_err = error[row_ind, col_ind].cpu().numpy()
                    sum_err_th += abs_err.sum()



            cnt_acc = cnt_acc_1 + cnt_acc_2 + cnt_acc_3 + cnt_acc_4
            sum_err = sum_err_1 + sum_err_2 + sum_err_3 + sum_err_4
            recall = num_acc / num_target
            precision = num_acc / num_pred
            F1 = 2 * recall * precision / (recall + precision)
            MAE = sum_err_th / num_pred
        
        
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
        print(f'Threshold method: recall {recall} precision {precision} F1 {F1} MAE {MAE}', flush=True)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
