import os
import numpy as np
import argparse

 

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('-c','--choices', 
                     default=300,type=int,
                   )
parser.add_argument('-r','--dir', 
                     default='/dataset/training',
                   )
args = parser.parse_args() 


def get_lists(folder,total):
	sub_list = os.listdir(folder)
	temp = []
	for sub_item in sub_list:
		if '.pnm' in sub_item:
			temp.append(folder+'/'+sub_item)
		elif '.' not in sub_item:
			get_lists(folder+'/'+sub_item,total)
	temp.sort(key=lambda x:int(x.split('/')[-1][:-4]))
	for item in temp:
		total.append(item)
	
total = []
#folder = '/data/test/'
folder = args.dir
get_lists(folder,total)

dict_ = {}
for i in total:
  dict_[i] = 1
total = []
for key in dict_:
  total.append(key)
choice_items = []
print('total',len(total))
for index in range(0,len(total)):
  if index != 0 and index%args.choices == 0:
	  choice_items.append(total[index])
csv = []
images = []
print(len(choice_items))

for item in choice_items:
  #print(item)
  item = '/'.join(item.split('/')[3:])
  #print(item)
  item = '.'.join(item.split('.')[:-1]) + '.jpg'
  csv.append(','.join([item,'0','0','0','0','sign']))
with open('/root/wangxin/test/test.csv','w') as f:
    f.write('\n'.join(csv))
with open('/root/wangxin/test/total.csv','w') as f:
	for item in total:
		item = '/'.join(item.split('/')[3:])
		item = '.'.join(item.split('.')[:-1]) + '.jpg'
		f.write(item+'\n')


