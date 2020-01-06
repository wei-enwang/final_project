import os
import numpy as np
from PIL import Image

IMAGE_DIRECTORY = "./JPP_mod/output/parsing/val"
POSE_DIRECTORY = "./JPP_mod/output/parsing/val"
JOINTS_DIRECTORY = "./JPP_mod/output/pose/val"

class trainset(Dataset):

    def __init__(self, transform):

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

        self.transform = transform

    def __getitem__(self, index):

        data = self.images[index]
        target = self.target[index]
        # load dataset and target images and transform them
        oi = Image.open(IMAGE_DIRECTORY+"/"+data+".jpg").convert('RGB')
        oip = Image.open(POSE_DIRECTORY+"/"+data+".png").convert('RGB')
        
        ti = Image.open(IMAGE_DIRECTORY+"/"+target+".jpg").convert('RGB')
        tip = Image.open(POSE_DIRECTORY+"/"+target+".png").convert('RGB')

        original_img = self.transform(oi)
        original_img_pose = self.transform(oip)
        target_img = self.transform(ti)
        target_img_pose = self.transform(tip)

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
        original_img_joints = self.transform(originalJoints)
        target_img_joints = self.transform(targetJoints)

        return original_img, original_img_pose, original_img_joints, target_img, target_img_pose, target_img_joints

    def __len__(self):
        return len(self.images)