# hash table
# create hash table:
import time
import os
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
#folder = '/dataset'
folder = '/dataset'
get_lists(folder,total)

def hash_create(folder,total):
	get_lists(folder,total)
	dict_ = {}
	for key in total:
		key = '/'.join(key.split('/')[-2:])
		dict_[key] = 1
	return dict_
hash_tabel = hash_create(folder,total)
print(hash_tabel)
t1 = time.time()
for key in total:
	key = '/'.join(key.split('/')[-2:])
	if key in hash_tabel:
		continue
t2 = time.time()
print(t2-t1)