from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
list1=[]
list2=[]
dic={}
for i in os.listdir('./JPP_mod/output/parsing/val'):
    if i[0:4] not in dic:
        dic[i[0:4]]=1
    else:
        dic[i[0:4]]+=1
for i in dic:
    for x in range(dic[i]):
        for j in range(dic[i]-1):
            list1.append(i+'_'+str(x)+'_vis')
        for w in range(dic[i]):
            if w==x:
                continue
            else:
                list2.append(i+'_'+str(w)+'_vis')
def default_loader(path):
    img_pil =  Image.open(path)
    img_pil = img_pil.resize((224,224))
    img_tensor = preprocess(img_pil)
    return img_tensor


#当然出来的时候已经全都变成了tensor
class trainset(Dataset):
    def __init__(self, loader=default_loader):
        #定义好 image 的路径
        self.list1 = list1
        self.list2 = list2
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img,target

    def __len__(self):
        return len(self.images)
