import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import TrainSetSmall
import models
import Visualizer

def load_checkpoint(g_path, d_path):

    discriminator_checkpoint = torch.load(d_path)
    generator_checkpoint = torch.load(g_path)

    d_optimizer = torch.optim.Adam(D.parameters(), lr = 3e-4)
    g_optimizer = torch.optim.Adam(G.parameters(), lr = 3e-4)

    D = models.discriminator()
    G = models.generator()

    d_optimizer.load_state_dict(discriminator_checkpoint['optimizer_state_dict'])
    g_optimizer.load_state_dict(generator_checkpoint['optimizer_state_dict'])

    D.load_state_dict(discriminator_checkpoint['model_state_dict'])
    G.load_state_dict(generator_checkpoint['model_state_dict'])

    D.train()
    G.train()

    assert discriminator_checkpoint['epoch'] == generator_checkpoint['epoch'], 'epoch number loading error'
    current_epoch = discriminator_checkpoint['epoch'] + 1

    return D, G, d_optimizer, g_optimizer, current_epoch
    
    

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type = int, default = 128)
parser.add_argument("--n_epochs", type = int, default = 3)
parser.add_argument("--mode", type = str, default = 'train')
args = parser.parse_args()
print(args)


print('start loading datasets...')
transformRGB = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])
transformGrey = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = (0.5,), std = (0.5,))])

train_data = TrainSetSmall.trainset(transformRGB = transformRGB, transformGrey = transformGrey)
print('number of data points: ', len(train_data))

trainloader = DataLoader(dataset = train_data, batch_size = args.batch_size, shuffle = False)

print('datasets loading finished!')

current_epoch = 0

if os.path.exists('./checkpoints/generator_checkpoint.pth') and os.path.exists('./checkpoints/discriminator_checkpoint.pth'):
    g_path = './checkpoints/generator_checkpoint.pth'
    d_path = './checkpoints/discriminator_checkpoint.pth'
    D, G, d_optimizer, g_optimizer, current_epoch = load_checkpoint(g_path, d_path)
    
else:
    D = models.discriminator()
    G = models.generator()

    d_optimizer = torch.optim.Adam(D.parameters(), lr = 3e-4)
    g_optimizer = torch.optim.Adam(G.parameters(), lr = 3e-4)

if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()

criterion = nn.BCELoss()

#start training
if current_epoch >= args.n_epochs:
    raise Exception('training has finished!')

print('start training!')

for epoch in range(current_epoch, args.n_epochs):
    for i, (original_img, original_img_pose, original_img_joints, target_img, target_img_pose, target_img_joints) in enumerate(trainloader):
        print(i)
        assert original_img.size() == original_img_pose.size() == target_img.size() == target_img_pose.size() and original_img_joints.size() == target_img_joints.size(), "data-points size error"
        
        num_img = original_img.size(0)#batch_size
        #img.size  = [B, C, H, W]

        '''# flatten the images
        original_img = original_img.view(num_img, -1)
        original_img_pose = original_img_pose.view(num_img, -1)
        original_img_joints = original_img_joints.view(num_img, -1)
        
        target_img = target_img.view(num_img, -1)
        target_img_pose = target_img_pose.view(num_img, -1)
        target_img_joints = target_img_joints.view(num_img, -1)'''

        ##train discriminator
        
        if torch.cuda.is_available():
            target_img = Variable(target_img).cuda()
            real_label = Variable(torch.ones(num_img)).cuda()
            fake_label = Variable(torch.zeros(num_img)).cuda()
        else:
            target_img = Variable(target_img)
            real_label = Variable(torch.ones(num_img))
            fake_label = Variable(torch.zeros(num_img))
            
        #compute loss of real images
        real_out = D(target_img)
        real_out = torch.squeeze(real_out)
        d_loss_real = criterion(real_out, real_label)
        real_scores = real_out #closer to 1 means better

        #compute loss of fake images
        z = torch.cat((original_img, target_img_joints), 1)
        if torch.cuda.is_available():
            z = z.cuda()

        fake_img = G(z)
        fake_img_original, fake_img_pose = fake_img.split([3, 1], dim = 1)
        #print('a:', fake_img_original.size())
        #print('b:', fake_img_pose.size())
        fake_out = D(fake_img_original)
        fake_out = torch.squeeze(fake_out)
        d_loss_fake = criterion(fake_out, fake_label)
        fake_scores = fake_out #closer to 0 means better

        #back-propagation and optimization
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward(retain_graph = True)
        d_optimizer.step()


        ##train generator
        
        #compute loss of fake images
        g_loss = criterion(fake_out, real_label)
        
        #back-propagation and optimization
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 100 == 0:
            print("iteration: {} / {}, Epoch: {} / {}, d_loss: {:.6f}, g_loss: {:.6f}, D real: {:.6f}, D fake: {:.6f}".format(
                str((i+1)*args.batch_size), str(len(train_data)), epoch, args.n_epochs, d_loss.data, g_loss.data, real_scores.data.mean(), fake_scores.data.mean()))

        
    fake_img_original = Visualizer.denorm(fake_img_original.cpu().data)
    save_image(fake_img_original, './img/fake_images_original-{}.png'.format(epoch+1))

    torch.save({'epoch': finished_epoch, 'model_state_dict': G.state_dict(), 'optimizer_state_dict': g_optimizer.state_dict()}, './checkpoints/generator_checkpoint.pth')
    torch.save({'epoch': finished_epoch, 'model_state_dict': D.state_dict(), 'optimizer_state_dict': d_optimizer.state_dict()}, './checkpoints/iscriminator_checkpoint.pth')