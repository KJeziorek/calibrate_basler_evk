# Code for calibrating the Basler and Event Camera

This code presents a pipeline for calibrating the Basler and Event Camera. The pipeline consists of the following steps:

1. Reconstructing the event camera data to grayscale images
2. Calibrating both cameras using OpenCV
3. Visualizing the results

The code is based on the E2VID method, which is a recurrent network for reconstructing event camera data to grayscale images. 

<p align="center">
  <img src="assets/calibrated.gif">
</p>

<p align="center">
  <img src="assets/ev+gray.gif">
</p>


# Installation

First, create a Conda environment:
    
```bash
conda create -n calib python=3.9
conda activate calib

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pandas scipy
conda install -c conda-forge opencv
conda install -y h5py blosc-hdf5-plugin -c conda-forge

```

Then, clone this repository and download the pretrained model for the E2VID algorithm:

```bash
git clone ...
wget "http://rpg.ifi.uzh.ch/data/E2VID/models/E2VID_lightweight.pth.tar" -O pretrained/E2VID_lightweight.pth.tar
```
Number of samples determines how many samples are used for calibration. The more samples, the better the calibration, however, the calibration process will take longer.

# Run

Start by exporting plugin for HDF5:

```bash
export HDF5_PLUGIN_PATH=$HDF5_PLUGIN_PATH:/usr/lib/x86_64-linux-gnu/hdf5/plugins 

```

Then you can run separate scripts for reconstructing the event camera data to grayscale images, calibrating both cameras and visualizing the results.

```bash
python3 reconstruct.py -i events/output.hdf5 --num_samples 100
python3 calibrate.py --num_samples 100
python3 reconstruct.py -i events/output.hdf5 --num_samples 100
```

or run the whole pipeline:

```bash
bash run_all.sh
```

Note that the reading events from HDF5 file is slow, so you need to be patient.

# Citation
This code is mainly based on the following paper:

```bibtex
@Article{Rebecq19cvpr,
  author        = {Henri Rebecq and Ren{\'{e}} Ranftl and Vladlen Koltun and Davide Scaramuzza},
  title         = {Events-to-Video: Bringing Modern Computer Vision to Event Cameras},
  journal       = {{IEEE} Conf. Comput. Vis. Pattern Recog. (CVPR)},
  year          = 2019
}
```
