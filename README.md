# Combination of pose parsing network and GAN

計程期末專題

> 把進度和目標寫在這

## Related resources

> pre-trained model: https://github.com/Engineering-Course/LIP_JPPNet <br>
> multi-person: https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation

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
- [ ] 找到可以抓出照片裡的人的 pretrained model <br>
- [ ] 將主角結合動作

## How to use our code?

#### JPPNet

In our case, **$(working_directory) = /JPP_mod**
1. Download pretrained model from[JPPNet google drive](https://drive.google.com/file/d/1BFVXgeln-bek8TCbRjN6utPAgRE0LJZg/view) and place it under **$(working_directory)/checkpoint/**
2. The images targeted for operation must be under **$(working_directory)/datasets/examples/image/**
3. Modify **$(working_directory)/datasets/examples/list/val.txt** to include the directories of the images(e.g. **/image/hello_world_man.jpg**).
4. Go to file **valuate_pose_JPPNet-s2.py** and **evaluate_parsing_JPPNet-s2.py** and change variable `NUM_STEPS` to the number of images under **$(working_directory)/datasets/examples/image/**
5. Run `evaluate_pose_JPPNet-s2.py` and `evaluate_parsing_JPPNet-s2.py` for pose estimation and body parts parsing respectively.
6. Results will be shown under **$(working_directory)/output/**


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
