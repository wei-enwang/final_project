import wx
import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import Visualizer
import models
import shutil
x=0
app = wx.PySimpleApp()
wildcard = " pic|*.jepg|pic|*.bmp|pic|*.gif|pic|*.jpg|pic|*.png"
while(True):
    print("please select a picture that include target pose:")
    dialog = wx.FileDialog(None, "Choose a file", os.getcwd(), "", wildcard, wx.FD_OPEN)
    if dialog.ShowModal() == wx.ID_OK:
        file1path=dialog.GetPath()
        print("picture selected!")
        x=1
    elif dialog.ShowModal()==wx.ID_CANCEL:
        print("please choose a picture!")
    dialog.Destroy()
    if x==1:
        break
x=0
while(True):
    print("please select a picture that include the person you want to perform the pose:")
    dialog = wx.FileDialog(None, "Choose a file", os.getcwd(), "", wildcard, wx.FD_OPEN)
    if dialog.ShowModal() == wx.ID_OK:
        file2path=dialog.GetPath()
        print("picture selected!")
        x=1
    elif dialog.ShowModal()==wx.ID_CANCEL:
        print("please choose a picture!")
    dialog.Destroy()
    if x==1:
        break
transformRGB = transforms.Compose([transforms.ToTensor(), transforms.Normalize(std = (0.5, 0.5, 0.5), mean = (0.5, 0.5, 0.5))])
transformGrey = transforms.Compose([transforms.ToTensor(), transforms.Normalize(std = (0.5,), mean = (0.5,))])
img1 = Image.open(file1path).convert('RGB')
img2 = Image.open(file2path).convert('RGB')
img1 = img1.resize((64, 128),Image.ANTIALIAS)
img2 = img2.resize((64, 128),Image.ANTIALIAS)
img1.save('./target.png')
img2.save('./original.png')
original=Image.open('./original.png').convert('RGB')
target=Image.open('./target.png').convert('RGB')
original=transformRGB(original)
target=transformRGB(target)
def load_checkpoint(g_path, d_path):

    D = models.discriminator()
    G = models.generator()

    discriminator_checkpoint = torch.load(d_path, map_location=torch.device('cpu'))
    generator_checkpoint = torch.load(g_path, map_location=torch.device('cpu'))

    D.load_state_dict(discriminator_checkpoint['model_state_dict'])
    G.load_state_dict(generator_checkpoint['model_state_dict'])

    D.eval()
    G.eval()

    return D, G

print('loading pre-trained model...')

try:
    g_path = './checkpoints/generator_checkpoint.pth'
    d_path = './checkpoints/discriminator_checkpoint.pth'
    D, G = load_checkpoint(g_path, d_path)
    print('pre-trained model successfully loaded!')
except:
    raise Exception('pre-trained model not found')


print('loading and transforming demo images...')
source=["./target.png","./original.png"]
now=os.getcwd()
os.chdir('./datasets/examples')
shutil.rmtree('./images')
os.makedirs('images')
os.chdir(now)
destination='./datasets/examples/images'
for x in source:
    shutil.move(x,destination)
import evaluate_parsing_JPPNet_s2
import evaluate_pose_JPPNet_s2
shutil.rmtree('./demo')
os.makedirs('demo')
source=['./output/parsing/val/original_vis.png','./output/parsing/val/target_vis.png','./output/pose/val/original.txt','./output/pose/val/target.txt']
destination='./demo'
for i in source:
    shutil.move(i,destination)
original_pose = Image.open('./demo/original_vis.png').convert('RGB')
target_pose = Image.open('./demo/target_vis.png').convert('RGB')
original_pose = transformRGB(original_pose)
target_pose = transformRGB(target_pose)
oj = open('./demo/original.txt')
oj = oj.readline().split()
original_joints = np.zeros((128, 64))

for i in range(0, len(oj), 2):
    original_joints[int(oj[i+1])][int(oj[i])] = 255
        
tj = open('./demo/target.txt')
tj = tj.readline().split()
target_joints = np.zeros((128, 64)) 

for i in range(0, len(tj), 2):
    target_joints[int(tj[i+1])][int(tj[i])] = 255

original_joints = Image.fromarray(original_joints)
original_joints = transformGrey(original_joints)
        
target_joints = Image.fromarray(target_joints)
target_joints = transformGrey(target_joints)
print('images loaded and transformed')


print('generating images...')

x = torch.cat((original, target_pose), 0)
x = x.unsqueeze(0)
fake_img = G(x)
fake_img = fake_img.squeeze(0)
fake_img_original, fake_img_pose = fake_img.split([3,3], 0)
fake_img_original = Visualizer.denorm(fake_img_original.data)
save_image(fake_img_original, './demo/fake_img.png')

print('done')
