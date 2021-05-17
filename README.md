# MFSNet
This repository contains the official implementation of our paper titled "MFSNet: A Multi Focus Segmentation Network for Skin Lesion Segmentation".

Abstract: Segmentation is an important task for medical image analysis to identify and localize diseases, for monitoring morphological changes and extract discriminative features for further diagnosis. Skin cancer is one of the most common types of cancer in the world, and its early diagnosis is pivotal for the complete elimination of malignant tumours from the body. In this research, we develop an Artificial Intelligence (AI) based framework for supervised skin lesion segmentation employing the deep learning approach. The proposed framework, called MFSNet (Multi-Focus Segmentation Network), uses differently scaled feature maps for computing the final segmentation mask using raw input RGB images of skin lesions. In doing so, initially, the images are preprocessed to remove unwanted artefacts and noises. The MFSNet employs the Res2Net backbone, a recently proposed convolutional neural network (CNN), for obtaining deep features which are used in a Parallel Partial Decoder (PPD) module to get a global map of the segmentation mask. In different stages of the network, convolution features and multi-scale maps are used in two boundary attention (BA) modules and two reverse attention (RA) modules to generate the final segmentation output. MFSNet when evaluated on three publicly available datasets: PH^2, ISIC 2017 and HAM10000 outperforms state-of-the-art methods, justifying the reliability of the framework.

# Preprocessing
To run the script for inpainting, run the following using the command prompt:

`python inpaint.py --root "D:/inputs/" --destination "D:/images/"`

# Training the Network
Follow the directory structure as follows:

```
+-- data
|   +-- .
|   +-- train
|   |       +-- images
|   |       +-- masks
|   +-- test
|   |       +-- images
|   |       +-- masks
+-- train.py
+-- test.py
```

Run the following to train the MFSNet network:

`python train.py --train_path "data/train"`

Other available hyperparameters for training are as follows:
- `--epoch`: Number of epochs of training. Default = 100
- `--lr`: Learning Rate. Default = 1e-4
- `--batchsize`: Batch Size. Default = 20
- `--trainsize`: Size of Training images (to be resized). Default = 352
- `--clip`: Gradient Clipping Margin. Default = 0.5
- `--decay_rate-`: Learning rate decay. Default = 0.05
- `--decay_epoch`: Number of epochs after which Learning Rate needs to decay. Default = 25

After the training is complete, run the following to generate the predictions on the test images:

`python test.py --test_path "data/test"`

# Evaluating Performance
Run `eval/main.m` using MATLAB on the ground truth images and the predicted masks, to get the evaluation measures.
