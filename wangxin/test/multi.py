import cv2
import numpy as np
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
import os
from tqdm import tqdm
import time


import argparse
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
from multiprocessing import Process
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('-o','--folder', 
                     default='/data/imgs'
                   )

args = parser.parse_args() 

names = {}
save_folder = args.folder
if not os.path.exists(save_folder):
  os.mkdir(save_folder)
with open('test.csv', 'r') as f:
    r = f.readlines()
    for line in r:
        line = line.split(',')[0]
        path = line.split('/')
        folder = '/'.join(path[:-1])
        
        name = folder + '/' + path[-1].split('.')[0] + '.pnm'
        names[name] = 1
n = []
for key in names:
	n.append(key)
names = n
#print(names)
print('choice images number is :',len(names))
#print(names)


def get_lists(folder,total):
	sub_list = os.listdir(folder)
	temp = []
	for sub_item in sub_list:
		if '.jpg' in sub_item:
			temp.append(folder+'/'+sub_item)
		elif '.' not in sub_item:
			get_lists(folder+'/'+sub_item,total)
	temp.sort(key=lambda x:int(x.split('/')[-1][:-4]))
	for item in temp:
		total.append(item)
	
total = []
#folder = '/dataset'
folder = '/data/imgs'
get_lists(folder,total)
#print(total)
def hash_create(folder,total):
	get_lists(folder,total)
	dict_ = {}
	for key in total:
		key = '/'.join(key[1:].split('/')[-3:])
		key = key[:-4] + '.pnm'
		dict_[key] = 1
	return dict_
hash_tabel = hash_create(folder,total)















#print(hash_tabel)














def converter1(names,hash_tabel=hash_tabel):
    #print(len(names))
    print('start process...')
    for name in tqdm(names):
        pnm_file = name
        if pnm_file in hash_tabel:
            #print(pnm_file,'already in')
            continue
        #print(pnm_file)
        pnm = cv2.imread('/dataset/training/'+pnm_file,0)
        eq = cv2.equalizeHist(pnm)
        #print(name)
        #print(eq.shape)
        pnms_3 = demosaicing_CFA_Bayer_bilinear(eq)
        if not cv2.imwrite(save_folder +'/'+'.'.join(name.split('.')[:-1])+'.jpg',pnms_3):
            w = cv2.imwrite(save_folder +'/'+'.'.join(name.split('.')[:-1])+'.jpg',pnms_3)
        #print(w)
def converter(names,num = 1):
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
			if not os.path.exists(save_folder+'/'+'/'.join(name.split('/')[:-1])):
				os.mkdir(save_folder+'/'+'/'.join(name.split('/')[:-2]))
				os.mkdir(save_folder+'/'+'/'.join(name.split('/')[:-1]))
			cv2.imwrite(save_folder +'/'+'.'.join(name.split('.')[:-1])+'.jpg',image)
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
		if not os.path.exists(save_folder+'/'+'/'.join(name.split('/')[:-1])):
				os.mkdir(save_folder+'/'+'/'.join(name.split('/')[:-2]))
				os.mkdir(save_folder+'/'+'/'.join(name.split('/')[:-1]))
		cv2.imwrite(save_folder +'/'+'.'.join(name.split('.')[:-1])+'.jpg',image)
		#print(index/num)
print('--run multi process----')
t1 = time.time()
l = int(len(names)/3)
folder_ ={}
for name in names:
    folder_[save_folder+'/'+'/'.join(name.split('/')[:-1])] = 1
for key in folder_:
    if not os.path.exists(key):
        if not os.path.exists('/'.join(key.split('/')[:-1])):
            os.mkdir('/'.join(key.split('/')[:-1]))
        os.mkdir(key)
p1 = Process(target=converter1, args=(names[:l],))

p1.start()
p2 = Process(target=converter1, args=(names[l:2*l],))

p2.start()
p3 = Process(target=converter1, args=(names[2*l:],))

p3.start()

p1.join()
print(' process1 finished')
p2.join()
print(' process2 finished')
p3.join()
print(' process3 finished')

p1.close()
p2.close()
p3.close()
t2 = time.time()
print('cost time:',t2-t1)
def clear():
    for key,value in globals().items():
        #if callable(value) or value.__class__.__name__ == 'module':
        #    continue
        del globals()[key]
#clear()

print('finished')
#time.sleep(10)
print('wait over')