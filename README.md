# MFSNet
This repository contains the official implementation of our paper titled "[MFSNet: A Multi Focus Segmentation Network for Skin Lesion Segmentation](https://doi.org/10.1016/j.patcog.2022.108673)" published in Pattern Recognition, Elsevier.  
[[Preprint](https://arxiv.org/abs/2203.14341)]

## Preprocessing
To run the script for inpainting, run the following using the command prompt:

`python inpaint.py --root "D:/inputs/" --destination "D:/images/"`

## Training the Network
Follow the directory structure as follows:

```
+-- data
|   +-- .
|   +-- train
|   |   +-- images
|   |   +-- masks
|   +-- test
|   |   +-- images
|   |   +-- masks
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
- `--decay_rate`: Learning rate decay. Default = 0.05
- `--decay_epoch`: Number of epochs after which Learning Rate needs to decay. Default = 25

After the training is complete, run the following to generate the predictions on the test images:

`python test.py --test_path "data/test"`

## Evaluating Performance
Run `eval/main.m` using MATLAB on the ground truth images and the predicted masks, to get the evaluation measures.

# Citation
If you find this repository useful, please cite our work:
```
@article{basak2022mfsnet,
  title={MFSNet: A Multi Focus Segmentation Network for Skin Lesion Segmentation},
  author={Basak, Hritam and Kundu, Rohit and Sarkar, Ram},
  journal={Pattern Recognition},
  pages={108673},
  year={2022},
  publisher={Elsevier}
}
```
