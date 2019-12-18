# combination of pose parsing network and GAN

計程期末專題

> 把進度和目標寫在這

## 相關資源

> pre-trained model: https://github.com/Engineering-Course/LIP_JPPNet <br>
> 多人: https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation

### 可能遇到的問題

1. 跑 LIP_JPPNet:

- tensorflow 用 1.15(pip install tensorflow==1.15)
- 跑出`AttributeError: module 'scipy.misc' has no attribute 'imread'`的話跑

```
> sudo pip install --upgrade scipy
> pip install scipy==1.1.0
```

我測試過的檔案放在/JPP_mod 裡面

## TODO

- [x] 找到適合的抓出關節的 pretrained model 並實作 <br>
- [ ] 找到可以抓出照片裡的人的 pretrained model <br>
- [ ] 將主角結合動作

## Citation

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
