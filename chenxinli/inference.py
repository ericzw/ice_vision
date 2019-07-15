import torch.nn as nn
import numpy as np
from model import Net
from data import data_transforms,data_jitter_hue,data_jitter_brightness,data_jitter_saturation,data_jitter_contrast,data_rotate,data_hvflip,data_shear,data_translate,data_center,data_grayscale


import argparse
from tqdm import tqdm
import os
import PIL.Image as Image

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as datasets
import numpy as np
import time

parser = argparse.ArgumentParser(description='evaluation script')
parser.add_argument('--c', type=int, default=60,metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")

args = parser.parse_args()
print('- - - - - - step3 Class inference Begin - - - - - - ')
start = time.time()
model_class = Net()
fc2_features = model_class.fc2.in_features
model_class.fc2 = nn.Linear(fc2_features, 121)
model_class.load_state_dict(torch.load('./new_model/class_121.pth'))
model_class.cuda()
model_class.eval()
model_temporary = Net()
fc2_features = model_temporary.fc2.in_features
model_temporary.fc2 = nn.Linear(fc2_features, 2)
model_temporary.load_state_dict(torch.load('./new_model/temporary_2.pth'))
model_temporary.cuda()
model_temporary.eval()
model_data = Net()
fc2_features = model_data.fc2.in_features
model_data.fc2 = nn.Linear(fc2_features, 20)
model_data.load_state_dict(torch.load('./new_model/data_20.pth'))
model_data.cuda()
model_data.eval()

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

transforms = [data_transforms,data_jitter_hue,data_jitter_brightness,data_jitter_saturation,data_jitter_contrast,data_rotate,data_hvflip,data_shear,data_translate,data_center]

output_file = open('./out.tsv', "w",encoding = 'utf-8')
output_file.write("frame,xtl,ytl,xbr,ybr,class,temporary,data\n".replace(',','\t'))
test_dir = './test'

dataset = os.listdir('../../data/imgClassi/total-class/train/')
dataset.sort()
label_class = {}
for i in range(len(dataset)):
    label_class[i] = dataset[i].replace('nan','NA')
dataset = os.listdir('../../data/imgClassi/total-temporary/train/')
dataset.sort()
label_temporary = {}
for i in range(len(dataset)):
    label_temporary[i] = dataset[i].replace('False','false').replace('True','true')
dataset = os.listdir('../../data/imgClassi/total-data/train/')
dataset.sort()
label_data = {}
for i in range(len(dataset)):
    label_data[i] = dataset[i].replace('not_data','')

for f in tqdm((os.listdir(test_dir))):
    if 'jpg' in f:
        output_class = torch.zeros([1, 121], dtype=torch.float32).cuda()
        output_temporary = torch.zeros([1, 2], dtype=torch.float32).cuda()
        output_data = torch.zeros([1, 20], dtype=torch.float32).cuda()
        with torch.no_grad():
            for i in range(0,len(transforms)):
                data = transforms[i](pil_loader(test_dir + '/' + f))
                data = data.view(1, data.size(0), data.size(1), data.size(2))
                data = Variable(data).cuda()
                output_class = output_class.add(model_class(data))
                output_temporary = output_temporary.add(model_temporary(data))
                output_data = output_data.add(model_data(data))
                
            output_class.cpu().numpy()
            output_temporary.cpu().numpy()
            output_data.cpu().numpy()
            pred_class = output_class.data.max(1, keepdim=True)[1]
            pred_temporary = output_temporary.data.max(1, keepdim=True)[1]
            pred_data = output_data.data.max(1, keepdim=True)[1]
            #print(f[0:15])
            tmp = f.split('_')
            imgName = tmp[-4] + '_'+ tmp[-3] + '_' + tmp[-2] + '/' + tmp[-1].split('.')[0]
            xtl = tmp[0]
            ytl = tmp[1]
            xbr = tmp[2]
            ybr = tmp[3]
            output_file.write("%s,%s,%s,%s,%s,%s,%s,%s\n".replace(',','\t') % (imgName,xtl,ytl,xbr,ybr, label_class[int(pred_class)],label_temporary[int(pred_temporary)],label_data[int(pred_data)]))
output_file.close()


test = open('./out.tsv','r+')
total = open('../wangxin/test/total.csv','r+')

f_file = './final.tsv'
if os.path.isfile(f_file):
    os.remove(f_file)
    print('=>remove file {}'.format(f_file))
    
final = open('./final.tsv','w+')
print('=>create file {}'.format(f_file))
final.write("frame,xtl,ytl,xbr,ybr,class,temporary,data\n".replace(',','\t'))
test_lines = test.readlines()
k = 0
for line in total:
    if k<=args.c :
        #print(k)
        a = test_lines[1].split('\t')
        b = line.split('/')
        frame = b[0] + '_' + b[1] + '/' + b[2].split('.')[0]
        final.write("%s,%s,%s,%s,%s,%s,%s,%s".replace(',','\t') % (frame, a[1], a[2], a[3], a[4], a[5], a[6], a[7]))
    else:
        if (k-1)//args.c > len(test_lines)-1:
            break
        a = test_lines[(k-1)//args.c + 1].split('\t')
        # a = test_lines[1].split('\t')
        b = line.split('/')
        frame = b[0] + '_' + b[1] + '/' + b[2].split('.')[0]
        final.write("%s,%s,%s,%s,%s,%s,%s,%s".replace(',','\t') % (frame, a[1], a[2], a[3], a[4], a[5], a[6], a[7]))
    k += 1
    
final.close()
test.close()
total.close()

end = time.time()
print('=>outfile saved in /root/chenxinli/final.tsv')
print('=> step3 time cost {:.2f} s'.format(end-start) )
print('- - - - - - step3 Class inference Down - - - - - -')
print('\n')