import os
import collections
path='/Users/wudongyu/Desktop/計程專題/train_renamed/'
files=os.listdir(path)
count = {}
doneNum = 0
for i in files:
    if i[0] != '.':
        oldName = path + i
        identity = i[:4]
        if identity not in count:
            count[identity] = 0
        else:
            count[identity] += 1
        newName = path + identity + '_' + str(count[identity]) + '.jpg'
        os.rename(oldName,newName)
        doneNum += 1
        if doneNum % 250 == 0:
            print(doneNum)
print('numImages:', doneNum)
