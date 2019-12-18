# combination of pose parsing network and GAN
計程期末專題
> 把進度和目標寫在這

## 相關資源
> pre-trained model: https://github.com/Engineering-Course/LIP_JPPNet <br> 
> 多人: https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation

### 可能遇到的問題
1. 跑LIP_JPPNet:
- tensorflow用1.15(pip install tensorflow==1.15)
- 跑出`AttributeError: module 'scipy.misc' has no attribute 'imread'`的話跑
```
sudo pip install --upgrade scipy
```

## TODO
- [ ]  找到適合的抓出關節的pretrained model並實作 <br>
- [ ]  找到可以抓出照片裡的人的pretrained model <br>
- [ ]  將主角結合動作
