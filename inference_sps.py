import argparse
import os
import time
import numpy as np
import pickle
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

import prepare_multi_sources_data
import func
import models
import switch_neural_network
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader


device = torch.device('cpu')
# device = torch.device('cuda:0')


##### args #####
start_plot = 78 # start plotting after epoch XX

def infer_one_epoch(epoch: int, val_data_path, test_data_path, output_path, pth_path, test_or_val, version):
    infer_start = time.time()
    spsnet = models.SPSnet()

    spsnet.load_state_dict(torch.load(pth_path, map_location=device))
    # spsnet.load_state_dict(torch.load(pth_path))
    # spsnet.to(device)

    # Construct loss function and Optimizer.
    criterion_sps = torch.nn.MSELoss()

    val_data = DataLoader(prepare_multi_sources_data.SSLR_Dataset(val_data_path), batch_size=100,
                        shuffle=True, num_workers=0)  # val_data is a tuple: (batch_x, batch_y, batch_z)
    test_data = DataLoader(prepare_multi_sources_data.SSLR_Dataset(test_data_path), batch_size=100,
                        shuffle=True, num_workers=0)  # test_data is a tuple: (batch_x, batch_y, batch_z)

    model_save_path = os.path.dirname(output_path)                    


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
    epoch_loss = 0.
    sam_size = 0.

    pkl_data = []

    with torch.no_grad():
        spsnet.eval()
        for index, (batch_x, batch_y, batch_z) in enumerate(val_data):
            batch_x = batch_x.to(device) # batch_x.shape [B, 8, 7, 337]
            batch_y = batch_y.to(device) # batch_y.shape [B, 360]
            batch_z = batch_z.to(device) # batch_z.shape [B,]

            # batch_z.shape[0] = batch_size

            # 获得模型预测结果
            a_1, output = spsnet(batch_x) # output.shape [B, 360]
            val_loss = criterion_sps(output, batch_y) # averaged loss on batch_y
            with torch.no_grad():
                epoch_loss += batch_z.size(0) * val_loss.clone().detach().item()
                sam_size += batch_z.size(0)


            # Plot SPS
            if epoch > start_plot:
                line_pred, = plt.plot(output[0], c='b', label="pred_sps")
                line_label, = plt.plot(batch_y[0], c='r', label="label_sps")
                plt.title("Comparison between estimated and ground truth SPS")
                plt.xlabel("DOA")
                plt.ylabel("Likelihood")
                plt.ylim((0, 1))
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
                    # print(f'cnt_acc_multi {cnt_acc_multi} sum_err_multi {sum_err_multi} total_multi {total_multi}')
                else:
                    pred = torch.tensor(func.get_topk_doa(output[batch], num_sources), dtype=torch.int, device=device)
                    # print(f'multi-sources pred {pred}', flush=True)
                    # print(f'multi-sources label {label}', flush=True)
                    
                    error = func.angular_distance(pred.reshape([num_sources, 1]), label.reshape([1, num_sources]))
                    row_ind, col_ind = linear_sum_assignment(error.numpy())
                    abs_err = error[row_ind, col_ind].numpy()
                    # print(f'multi-sources error {error}', flush=True)
                    # print(f'multi-sources abs_err {abs_err}', flush=True)

                    if num_sources == 3:
                        cnt_acc_3 += np.sum(abs_err <= 5)
                        sum_err_3 += abs_err.sum()
                        total_3 += 3
                    if num_sources == 4:
                        cnt_acc_4 += np.sum(abs_err <= 5)
                        sum_err_4 += abs_err.sum()
                        total_4 += 4
                # Save SPS
                if test_or_val == "val":
                    output_path = output_path.replace("sps_train", "sps_val")
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                pspectrum = output[batch]
                sample_data = {"sps" : pspectrum, "num_sources" : num_sources}
                pkl_data.append(sample_data)

                while num_sources > 0:
                    num_sources -= 1
                    peak, pspectrum = func.pop_peak(pspectrum)
                    sample_data = {"sps" : pspectrum, "num_sources" : num_sources}
                    pkl_data.append(sample_data)



        sec_val_arr = torch.tensor(sec_val_list, dtype=torch.float)        
        # print(f'sec_val_arr[:100] {sec_val_arr[:100]}', flush=True)
        threshold_mean = torch.mean(sec_val_arr).item()
        threshold_std = torch.std(sec_val_arr).item()
        print(f'threshold_mean {threshold_mean} threshold_std {threshold_std}', flush=True)

        cnt_acc = cnt_acc_1 + cnt_acc_2 + cnt_acc_3 + cnt_acc_4
        sum_err = sum_err_1 + sum_err_2 + sum_err_3 + sum_err_4

        epoch_loss = epoch_loss / sam_size
    print(f'epoch {epoch + 1} epoch_loss {epoch_loss} sam_size {sam_size}', flush=True)
    print(f'epoch {epoch + 1}\'s validation stage, total_zero, {total_0}, total_1 {total_1} total_2 {total_2} total_3 {total_3} total_4 {total_4} total {total}', flush=True)
    print('========== Validation results for known number of sources ==========', flush=True)
    print('Single-source accuracy on val set: %.2f %% ' % (100.0 * cnt_acc_1 / total_1), flush=True)
    print('Single-source MAE on val set: %.3f ' % (sum_err_1 / total_1), flush=True)
    print('Two-sources accuracy on val set: %.2f %% ' % (100.0 * cnt_acc_2 / total_2), flush=True)
    print('Two-sources MAE on val set: %.3f ' % (sum_err_2 / total_2), flush=True)             
    print('Three-sources accuracy on val set: %.2f %% ' % (100.0 * cnt_acc_3 / total_3), flush=True)
    print('Three-sources MAE on val set: %.3f ' % (sum_err_3 / total_3), flush=True)   
    # print('Four-sources accuracy on val set: %.2f %% ' % (100.0 * cnt_acc_4 / total_4), flush=True)
    # print('Four-sources MAE on val set: %.3f ' % (sum_err_4 / total_4), flush=True)   
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
        spsnet.eval()
        for index, (batch_x, batch_y, batch_z) in enumerate(test_data):
            batch_x = batch_x.to(device) # batch_x.shape [B, 8, 7, 337]
            batch_y = batch_y.to(device) # batch_y.shape [B, 360]
            batch_z = batch_z.to(device) # batch_z.shape [B,]


            # 获得模型预测结果
            a_1, output = spsnet(batch_x) # output.shape [B, 360]

            for batch in range(batch_z.size(0)):
                # test for known number of sources
                num_sources = batch_z[batch].item()
                total += num_sources


                # Plot SPS
                if epoch > start_plot + 10:
                    line_pred, = plt.plot(output[batch], c='b', label="pred_sps")
                    line_label, = plt.plot(batch_y[batch], c='r', label="label_sps")
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
                    # print(f'multi-sources pred {pred}', flush=True)
                    # print(f'multi-sources label {label}', flush=True)
                    
                    error = func.angular_distance(pred.reshape([num_sources, 1]), label.reshape([1, num_sources]))
                    row_ind, col_ind = linear_sum_assignment(error.numpy())
                    abs_err = error[row_ind, col_ind].numpy()
                    # print(f'multi-sources error {error}', flush=True)
                    # print(f'multi-sources abs_err {abs_err}', flush=True)

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
                peaks = func.get_topk_doa_threshold(output[batch].detach(), threshold_mean - threshold_std)
                peaks = torch.tensor(peaks, dtype=torch.int, device=device)
                
                for l in label:
                    for i in range(l - 5, l + 6):
                        if i in peaks:
                            num_acc += 1
                num_pred += len(peaks)
                num_target += label.size(0)

                error = func.angular_distance(peaks.reshape([len(peaks), 1]), label.reshape([1, num_sources]))
                row_ind, col_ind = linear_sum_assignment(error.numpy())
                abs_err = error[row_ind, col_ind].numpy()
                sum_err_th += abs_err.sum()

                # Save SPS
                if test_or_val == "val":
                    output_path = output_path.replace("sps_val", "sps_test")
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                pspectrum = output[batch]
                sample_data = {"sps" : pspectrum, "num_sources" : num_sources}
                pkl_data.append(sample_data)

                # # Plot SPS
                # if num_sources == 2:
                #     line_pred, = plt.plot(pspectrum.cpu(), c='b')
                #     # line_label, = plt.plot(batch_y[0].cpu(), c='r', label="label_sps")
                #     # plt.title("Comparison between estimated and ground truth SPS")
                #     plt.xlabel("DOA")
                #     plt.ylabel("Likelihood")
                #     plt.ylim((0, 1))
                #     # plt.legend(handles=[line_pred])
                #     ax=plt.gca() # ax为两条坐标轴的实例
                #     ax.xaxis.set_major_locator(MultipleLocator(30)) # 把x轴的主刻度设置为30的倍数
                #     # ax.yaxis.set_major_locator(MultipleLocator(1)) # 把y轴的主刻度设置为1的倍数
                #     if not os.path.exists(model_save_path + '/pred_sps'):
                #         os.makedirs(model_save_path + '/pred_sps')
                #     plt.savefig(model_save_path + f'/pred_sps/test_{epoch + 1}th_epoch_{index}th_batch_{0}_nos_{num_sources}.png', dpi=600)
                #     plt.close()

                #     peak, new_pspectrum = func.pop_peak(pspectrum)
                #     line_pred, = plt.plot(new_pspectrum.cpu(), c='b')
                #     # line_label, = plt.plot(batch_y[0].cpu(), c='r', label="label_sps")
                #     # plt.title("Comparison between estimated and ground truth SPS")
                #     plt.xlabel("DOA")
                #     plt.ylabel("Likelihood")
                #     plt.ylim((0, 1))
                #     # plt.legend(handles=[line_pred])
                #     ax=plt.gca() # ax为两条坐标轴的实例
                #     ax.xaxis.set_major_locator(MultipleLocator(30)) # 把x轴的主刻度设置为30的倍数
                #     plt.savefig(model_save_path + f'/pred_sps/test_{epoch + 1}th_epoch_{index}th_batch_{0}_nos_{num_sources-1}.png', dpi=600)
                #     plt.close()

                #     peak, new_pspectrum = func.pop_peak(new_pspectrum)
                #     line_pred, = plt.plot(new_pspectrum.cpu(), c='b')
                #     # line_label, = plt.plot(batch_y[0].cpu(), c='r', label="label_sps")
                #     # plt.title("Comparison between estimated and ground truth SPS")
                #     plt.xlabel("DOA")
                #     plt.ylabel("Likelihood")
                #     plt.ylim((0, 1))
                #     # plt.legend(handles=[line_pred])
                #     ax=plt.gca() # ax为两条坐标轴的实例
                #     ax.xaxis.set_major_locator(MultipleLocator(30)) # 把x轴的主刻度设置为30的倍数
                #     plt.savefig(model_save_path + f'/pred_sps/test_{epoch + 1}th_epoch_{index}th_batch_{0}_nos_{num_sources-2}.png', dpi=600)
                #     plt.close()

                while num_sources > 0:
                    num_sources -= 1
                    peak, pspectrum = func.pop_peak(pspectrum)
                    sample_data = {"sps" : pspectrum, "num_sources" : num_sources}
                    pkl_data.append(sample_data)
       



        cnt_acc = cnt_acc_1 + cnt_acc_2 + cnt_acc_3 + cnt_acc_4
        sum_err = sum_err_1 + sum_err_2 + sum_err_3 + sum_err_4
        recall = num_acc / num_target
        precision = num_acc / num_pred
        F1 = 2 * recall * precision / (recall + precision)
        MAE = sum_err_th / num_pred
    
    sps_save_path = os.path.join(output_path, f'sps_test.pkl')
    # print(sps_save_path, flush=True)
    pkl_file = open(sps_save_path, 'wb')
    pickle.dump(pkl_data, pkl_file)
    pkl_file.close()
    print(f'epoch {epoch + 1}\'s test stage, total_zero, {total_0}, total_1 {total_1} total_2 {total_2} total_3 {total_3} total_4 {total_4} total {total}', flush=True)
    print('========== Test results on known number of sources ==========', flush=True)
    print('Single-source accuracy on test set: %.2f %% ' % (100.0 * cnt_acc_1 / total_1), flush=True)
    print('Single-source MAE on test set: %.3f ' % (sum_err_1 / total_1), flush=True)
    print('Two-sources accuracy on test set: %.2f %% ' % (100.0 * cnt_acc_2 / total_2), flush=True)
    print('Two-sources MAE on test set: %.3f ' % (sum_err_2 / total_2), flush=True)             
    print('Three-sources accuracy on test set: %.2f %% ' % (100.0 * cnt_acc_3 / total_3), flush=True)
    print('Three-sources MAE on test set: %.3f ' % (sum_err_3 / total_3), flush=True)   
    # print('Four-sources accuracy on test set: %.2f %% ' % (100.0 * cnt_acc_4 / total_4), flush=True)
    # print('Four-sources MAE on test set: %.3f ' % (sum_err_4 / total_4), flush=True)   
    print('Overall accuracy on test set: %.2f %% ' % (100.0 * cnt_acc / (total_1 + total_2 + total_3 + total_4)), flush=True)
    print('Overall MAE on test set: %.3f ' % (sum_err / (total_1 + total_2 + total_3 + total_4)), flush=True)
    print('===== Test results on unknown number of sources =====', flush=True)
    print(f'Threshold method: recall {recall} precision {precision} F1 {F1} MAE {MAE}', flush=True)
    print('===== Test results on the pred of number of sources =====', flush=True)

    torch.cuda.empty_cache()

    print('Time cost of inference stage %.2f'%(time.time()-infer_start), flush=True)



if __name__ == '__main__':
    val_data_path = "/Work18/2021/fuyanjie/exp_data/sim_audio_vctk/sim_val_data_frame_level"
    test_data_path = "/Work18/2021/fuyanjie/exp_data/sim_audio_vctk/sim_test_1_data"
    output_path = "/Work18/2021/fuyanjie/exp_data/exp_cnn_sc/CNN-SC-A1Wdata"
    pth_path = "/Work18/2021/fuyanjie/exp_data/exp_cnn_sc/CNN-SC-2Wdata/Second_Stage_Epoch59.pth"

    parser = argparse.ArgumentParser(description='inference stage')
    parser.add_argument('--epoch', metavar='EPOCH', type=int,
                        default=-1, help='index of epoch')
    parser.add_argument('--val_data_path', metavar='VAL_DATA_PATH', type=str,
                        default=val_data_path, help='path to the validation data')
    parser.add_argument('--test_data_path', metavar='TEST_DATA_PATH', type=str,
                        default=test_data_path, help='path to the test data')
    parser.add_argument('--output_path', metavar='OUTPUT_PATH', type=str,
                        default=output_path, help='path to the output folder')       
    parser.add_argument('--pth_path', metavar='PTH_PATH', type=str,
                        default=pth_path, help='path to the saved pth model')                    
    parser.add_argument('--test_or_val', metavar='TEST_OR_VAL', type=str,
                        default="test", help='test_or_val')       
    parser.add_argument('--version', metavar='VERSION', type=str,
                        default=1, help='version') 
    args = parser.parse_args()

    # Print arguments
    print('========== Print arguments ==========', flush=True)
    for k, v in vars(args).items():
        print(k,' = ',v, flush=True)
    print('========== Print arguments ==========', flush=True)
    infer_one_epoch(args.epoch, args.val_data_path, args.test_data_path, args.output_path, args.pth_path, args.test_or_val, args.version)
