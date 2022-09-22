#!/bin/bash
#$ -S /bin/bash

#here you'd best to change testjob as username
#$ -N inference_spsnet

#cwd define the work environment,files(username.o) will generate here
#$ -cwd

# merge stdo and stde to one file
#$ -j y

# resource requesting, e.g. for gpu use
#$ -l h=cpu01

# remember to activate your conda env
source activate torch1.8+cu111

test_or_val="test"
version=5

echo "job start time: `date`"
# start whatever your job below, e.g., python, matlab, etc.
#ADD YOUR COMMAND HERE,LIKE python3 main.py
                                           
# CUDA_VISIBLE_DEVICES=1 \
python ../inference_sps.py --epoch 1 \
                       --val_data_path "/Work21/2021/fuyanjie/exp_data/sim_audio_vctk/sim_3sources_${test_or_val}_${version}_data" \
                       --test_data_path "/Work21/2021/fuyanjie/exp_data/sim_audio_vctk/sim_3sources_${test_or_val}_$(($version+1))_data" \
                       --output_path "/Work21/2021/fuyanjie/exp_data/exp_sps_isc/3sources/sps_train" \
                       --pth_path "/Work21/2021/fuyanjie/exp_data/exp_sps_isc/SPS-3sources-0/SPSnet_Epoch23.pth" \
                       --test_or_val "val" \
                       --version ${version}


#chmod a+x run.sh

# hostname
# sleep 10
echo "job end time:`date`"
