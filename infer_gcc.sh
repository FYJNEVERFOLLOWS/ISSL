#!/bin/bash
#$ -S /bin/bash

model="SPS-SC-50rooms4sources"

#here you'd best to change testjob as username
#$ -N "inference_4gcc"

#cwd define the work environment,files(username.o) will generate here
#$ -cwd

# merge stdo and stde to one file
#$ -j y

# resource requesting, e.g. for gpu use
#$ -l h=cpu01

# remember to activate your conda env
source activate torch1.8+cu111

echo "job start time: `date`"
# start whatever your job below, e.g., python, matlab, etc.
#ADD YOUR COMMAND HERE,LIKE python3 main.py
                                            
# CUDA_VISIBLE_DEVICES=2 \
python ../inference_gcc.py --epoch -1 \
                       --val_data_path "/Work21/2021/fuyanjie/exp_data/sim_audio_vctk/sim_4sources_test_6_gcc" \
                       --test_data_path "/Work21/2021/fuyanjie/exp_data/sim_audio_vctk/sim_4sources_test_5_gcc" \
                       --model_save_path "/Work21/2021/fuyanjie/exp_data/exp_sps_isc/4->ISC-4sources2W-0" \
                       --pth_path "/Work21/2021/fuyanjie/exp_data/exp_sps_isc/GCC-4sources-0/SPSnet_Epoch11.pth" \
                       --sc_pth_path "/Work21/2021/fuyanjie/exp_data/exp_sps_isc/SC-4sources-2W-0/SCnet_Epoch17.pth" \
                       --isc_pth_path "/Work21/2021/fuyanjie/exp_data/exp_sps_isc/4->ISC-4sources2W-0/ISCnet_Epoch37.pth"
                     #   --isc_pth_path "/Work21/2021/fuyanjie/exp_data/exp_sps_isc/3->ISC-3sources2W-0/ISCnet_Epoch13.pth"
                                                

                    #    --pth_path "/Work21/2021/fuyanjie/exp_data/exp_sps_sc/SPS-SIM50rooms-2/SPSnet_Epoch16.pth" \

#chmod a+x run.sh

# hostname
# sleep 10
echo "job end time:`date`"
