import cv2
import numpy as np
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
import os
# historgram
# pnm_file = ''
# pnm = cv2.imread(pnm_file)
# eq = cv2.equalizeHist(pnm)

# # convert to 3 channel
# cv2.imwrite('root/wangxin/eq.jpg',demosaicing_CFA_Bayer_Menon2007(eq))

names = []
save_folder = '/root/wangxin/test/'
with open('test_dataset.csv', 'r') as f:
    r = f.readlines()
    for line in r:
        line = line.split(',')[0]
        path = line.split('/')
        folder = path[0].split('_')
        folder = '_'.join(folder[:-1]) + '/' + folder[-1]
        name = folder + '/' + path[-1].split('.')[0] + '.pnm'
        names.append(name)
for index,name in enumerate(names):
    pnm_file = name
    pnm = cv2.imread('/dataset/training/'+pnm_file,0)
    eq = cv2.equalizeHist(pnm)

    # convert to 3 channel
    #check folder 
    if not os.path.exists(save_folder+'/'.join(name.split('/')[:-1])):
        os.mkdir('/'.join(name.split('/')[:-2]))
        os.mkdir('/'.join(name.split('/')[:-1]))
    #print(eq.shape)
    print(cv2.imwrite(save_folder +'.'.join(name.split('.')[:-1])+'.jpg',demosaicing_CFA_Bayer_Menon2007(eq)))
    print(index/len(names))