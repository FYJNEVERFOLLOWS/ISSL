# ISSL

The official implementation for the paper ["Iterative Sound Source Localization for Unknown Number of Sources"](https://arxiv.org/abs/2206.12273).

## Dataset

See https://github.com/FYJNEVERFOLLOWS/VCTK-SIM



Dependency
----------

* `torch+torchaudio <https://pytorch.org/>`

* `apkit <https://github.com/hwp/apkit>`_ (version 0.2)

* `scikit-learn`

* `scipy`

  

## Training

1. Run train_ssnet.py and save a checkpoint of SSNet;
2. Load the saved SSNet's path and run inference_sps.py on another training set to build the spatial spectrum dataset;
3. Run train_asdnet.py on the spatial spectrum dataset and save a checkpoint of ASDNet;
4. Load both SSNet and ASDNet, run inference.py to test DOA estimation performance on unknown multiple sources. 
