import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

IMAGE_DIRECTORY = "./datasets/train_original"
POSE_DIRECTORY = "./datasets/train_pose"
JOINTS_DIRECTORY = "./datasets/train_joint"

class trainset(Dataset):

    def __init__(self, transformRGB, transformGrey):

        self.transformRGB = transformRGB
        self.transformGrey = transformGrey

        # create dataset lists for training
        self.images = []
        self.target = []
        imageDict = {}
        for image in os.listdir(IMAGE_DIRECTORY):
            image_name = image[:-4]
            try:
                if image_name[4] == "_":
                    original_id = image_name[0:4]
                     
                    if original_id in imageDict:
                        imageDict[original_id] += 1
                    # change the line below to "else: " to train the entire dataset 
                    elif original_id < "0099":
                        imageDict[original_id] = 1
            except IndexError:
                continue
        for image in imageDict:
            for subject in range(imageDict[image]):
                for target in range(imageDict[image]):
                    if subject == target:
                        continue
                    else:
                        self.images.append(image+"_"+str(subject))
                        self.target.append(image+"_"+str(target))


    def __getitem__(self, index):

        data = self.images[index]
        target = self.target[index]
        # load dataset and target images and transform them
        oi = Image.open(IMAGE_DIRECTORY+"/"+data+".jpg").convert('RGB')
        oip = Image.open(POSE_DIRECTORY+"/"+data+".png").convert('RGB')
        
        ti = Image.open(IMAGE_DIRECTORY+"/"+target+".jpg").convert('RGB')
        tip = Image.open(POSE_DIRECTORY+"/"+target+".png").convert('RGB')

        oi = self.transformRGB(oi)
        oip = self.transformRGB(oip)
        ti = self.transformRGB(ti)
        tip = self.transformRGB(tip)

        # convert joints txt file into trainable data
        oJointsFile = open(JOINTS_DIRECTORY+"/"+data+".txt")
        oJoints = oJointsFile.readline().split()
        originalJoints = np.zeros((128, 64))

        for i in range(0, len(oJoints), 2):
            originalJoints[int(oJoints[i+1])][int(oJoints[i])] = 255
        
        tJointsFile = open(JOINTS_DIRECTORY+"/"+target+".txt")
        tJoints = tJointsFile.readline().split()
        targetJoints = np.zeros((128, 64)) 

        for i in range(0, len(tJoints), 2):
            targetJoints[int(tJoints[i+1])][int(tJoints[i])] = 255

        # transform joints array
        originalJoints = Image.fromarray(originalJoints)
        originalJoints = self.transformGrey(originalJoints)
        
        targetJoints = Image.fromarray(targetJoints)
        targetJoints = self.transformGrey(targetJoints)

        return oi, oip, originalJoints, ti, tip, targetJoints

    def __len__(self):
        return len(self.images)
