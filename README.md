# LuNG3D

# Synthetic Lung Nodule 3D Image Generation Using Autoencoders
This repository contains the autoencoder and lung nodule analyzer used in "Synthetic Lung Nodule 3D Image Generation Using Autoencoders"(https://arxiv.org/abs/1811.07999).

This code is expected to be usable with Python 3.6.5 and PyTorch 0.4.1, but this code is for example reference only and is not maintained.

The majority of the analyzer code in myALNSB is cloned from "ALNSB: the Adaptive Lung Nodule Screening Benchmark"(https://github.com/sincewhenUCLA/Lung-Nodule-Detection-C-Benchmark).

## LuNG
The python scripts below are in LuNG:
* Shapes.py
  Primary Python code which loads in 3D training images, defines autoencoder,
  trains model, and creates generated images.
* nofeed6.py
  Runs 6 training iteration blocks allowing interaction with ALNSB for 
  evaluation. Does not use ALNSB-approved nodules for training augmentation.
* feedtwice.py
  Runs 6 training iteration blocks allowing interaction with ALNSB for 
  evaluation and training augmentation. Augments 2 of the 6 iteration blocks.

## myALNSB
The scripts below are in myALNSB/Lung-Nodule-Detection-C-Benchmark:
* run4Single.sh
  - Run 4 model directories in parallel.
  - Waits for LuNG Python generation scripts to produce rnd.csv file.
  - Creates segInputOverride.csv which is used by alnsb for nodule processing 
    Normally, alnsb processes the raw image which is in images/NLST_R0960B_OUT4
  - Creates img2.csv (possible training augmentation) and feat.csv
    (analyzer evaluation of nodules). 
* run64Single.sh runs 6 iteration of run4Single to interact with nofeed6.py
  or feedtwice.py autoencoder training and generation.
In myALNSB/Lung-Nodule-Dection-C-Benchmark/stages/classification:
* classification_step.c: processes nodule for classifier step and generates
  accepted images with KEEPCSV tokens and feature data with KEEPFEET tokens.

## Author

Steve Kommrusch is the author of the LuNG autoencoder code, changes to the ALNSB analizer, summary, and processing scripts.

# LuNG3D
