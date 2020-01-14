# Combination of pose parsing network and GAN

計程期末專題

> 把進度和目標寫在這

## Introduction
In this project, we implement [Joint Body Parsing & Pose Estimation Network (JPPNet)](https://github.com/Engineering-Course/LIP_JPPNet) and GAN to generate images of an individual showing different poses. Our work is primarily established upon [PyTorch](https://pytorch.org/).

First, we use the pretrained JPPNet model to produce training datasets for our GAN network. JPPNet parses an image and return an image of labeled different body parts and a text file containing the joints' position. Our network receives two images of the same person along with the data produced by JPPNet. One being the input and another being the target.

## Related resources

> pre-trained model: https://github.com/Engineering-Course/LIP_JPPNet <br>
> multi-person: https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation <br>
> turn pose into body(MV): https://github.com/xrenaa/Music-Dance-Video-Synthesis <br>
> Introduction of market1501 dataset: http://blog.fcj.one/reid-market-1501.html

### Common problems running this code

#### JPPNet:

- we use tensorflow 1.15 here (pip install tensorflow==1.15).
- when `AttributeError: module 'scipy.misc' has no attribute 'imread'` occurs run

```
> sudo pip install --upgrade scipy
> pip install scipy==1.1.0
```

我測試過的檔案放在/JPP_mod 裡面，直接用這個就好。

## TODO

- [x] 找到適合的抓出關節的 pretrained model 並實作 <br>
- [x] 做好可以抓出照片裡的人的 pretrained model <br>
- [x] 用 JPP_mod 生成 dataset <br>
- [x] 將生成的 dataset 用 rename_img.py 改名 <br>
- [x] 用 JPP_mod 對照片 dataset 生成的資料訓練 model <br>
- [x] 將主角結合動作

## How to use our code?
**\$important!!/**
Recommandation:git clone this project and it will be much easier to finish!!
####Our Model
You need to download these two files and place it under checkpoints(at easydisplay)
https://drive.google.com/file/d/18-kL51Qf2SKSMROJ80fOPWfPu_wkRKDh/view?usp=sharing
https://drive.google.com/file/d/1AW3kWXnt7-y4ZsVsF-gWEGYCn5WwQXn6/view?usp=sharing

#### JPPNet

> In our case, **\$(working_directory) = /JPP_mod**

1. Download pretrained model from[JPPNet google drive](https://drive.google.com/file/d/1BFVXgeln-bek8TCbRjN6utPAgRE0LJZg/view) and place it under **\$(working_directory)/checkpoint/**(There are two checkpoint now:one in JPP_mod,one in easydisplay)
2. The images targeted for operation must be under **\$(working_directory)/datasets/examples/image/**
3. Go to file **valuate_pose_JPPNet-s2.py** and **evaluate_parsing_JPPNet-s2.py** and change variable `NUM_STEPS` to the number of images under **\$(working_directory)/datasets/examples/image/**
4. Run `evaluate_pose_JPPNet-s2.py` and `evaluate_parsing_JPPNet-s2.py` for pose estimation and body parts parsing respectively.
5. Results will be shown under **\$(working_directory)/output/**


## Reference

```
@article{liang2018look,
  title={Look into Person: Joint Body Parsing \& Pose Estimation Network and a New Benchmark},
  author={Liang, Xiaodan and Gong, Ke and Shen, Xiaohui and Lin, Liang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2018},
  publisher={IEEE}
}

@InProceedings{Gong_2017_CVPR,
  author = {Gong, Ke and Liang, Xiaodan and Zhang, Dongyu and Shen, Xiaohui and Lin, Liang},
  title = {Look Into Person: Self-Supervised Structure-Sensitive Learning and a New Benchmark for Human Parsing},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {July},
  year = {2017}
}
```
