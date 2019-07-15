import cv2
import numpy as np
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
import os
from tqdm import tqdm
import time
from colour_demosaicing import (
    
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_bilinear,
    mosaicing_CFA_Bayer)
# historgram
# pnm_file = ''
# pnm = cv2.imread(pnm_file)
# eq = cv2.equalizeHist(pnm)

# # convert to 3 channel
# cv2.imwrite('root/wangxin/eq.jpg',demosaicing_CFA_Bayer_bilinear(eq))


names = {}
save_folder = '/root/wangxin/test/'
with open('test_dataset.csv', 'r') as f:
    r = f.readlines()
    for line in r:
        line = line.split(',')[0]
        path = line.split('/')
        folder = path[0].split('_')
        folder = '_'.join(folder[:-1]) + '/' + folder[-1]
        name = folder + '/' + path[-1].split('.')[0] + '.pnm'
        names[name] = 1
n = []
for key in names:
	n.append(key)
names = n
print(len(names))
#print(names)
num = 10
t1 = time.time()
for epoch in tqdm(range(int(len(names)/num))):

	pnms = np.zeros([2048,2448*num])
	for index,name in enumerate(names[epoch*num:(epoch+1)*num]):
		pnm_file = name
		pnm = cv2.imread('/dataset/training/'+pnm_file,0)
		eq = cv2.equalizeHist(pnm)
		#print(pnm.shape)
		pnms[:,index*2448:(index+1)*2448] = eq
	#print('read over')

	#print('eq over')
	pnms_3 = demosaicing_CFA_Bayer_bilinear(pnms)
	#print('demosaicing_CFA_Bayer_bilinear over')
	#print('write')
	for index,name in enumerate(names[epoch*num:(epoch+1)*num]):
		#print(pnms_3[:,index*2448:index*2448+2448,:].shape)
		image = pnms_3[:,index*2448:index*2448+2448,:] 
		#print(image.shape)
		if not os.path.exists(save_folder+'/'.join(name.split('/')[:-1])):
			os.mkdir(save_folder+'/'.join(name.split('/')[:-2]))
			os.mkdir(save_folder+'/'.join(name.split('/')[:-1]))
		cv2.imwrite(save_folder +'.'.join(name.split('.')[:-1])+'.jpg',image)
		#print(index/num)
pnms = np.zeros([2048,2448*num])
for index,name in enumerate(names[(epoch+1)*num:]):
	pnm_file = name
	pnm = cv2.imread('/dataset/training/'+pnm_file,0)
	eq = cv2.equalizeHist(pnm)
	#print(pnm.shape)
	pnms[:,index*2448:(index+1)*2448] = eq
print('read over')

print('eq over')
pnms_3 = demosaicing_CFA_Bayer_bilinear(pnms)
#print('demosaicing_CFA_Bayer_bilinear over')
#print('write')
for index,name in enumerate(names[(epoch+1)*num:]):
	image = pnms_3[:,index*2448:(index+1)*2448,:] 
	if not os.path.exists(save_folder+'/'.join(name.split('/')[:-1])):
			os.mkdir(save_folder+'/'.join(name.split('/')[:-2]))
			os.mkdir(save_folder+'/'.join(name.split('/')[:-1]))
	cv2.imwrite(save_folder +'.'.join(name.split('.')[:-1])+'.jpg',image)
	#print(index/num)
t2 = time.time()
print(t2-t1,'finished')
		