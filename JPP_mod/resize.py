from PIL import Image

path1 = './datasets/examples/images/big.jpg'
path2 = './datasets/examples/images/spider.jpg'

img1 = Image.open(path1).convert('RGB')
img2 = Image.open(path2).convert('RGB')



img1 = img1.resize((64, 128),Image.ANTIALIAS)
img2 = img2.resize((64, 128),Image.ANTIALIAS)

img1.save('./demo_target.jpg')
img2.save('./demo_original.jpg')
