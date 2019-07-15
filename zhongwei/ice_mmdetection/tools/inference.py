from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import os
import json
import numpy as np
import time

def main():
    config_file = 'configs/cascade_rcnn_x101_64x4d_fpn_1x.py'
    checkpoint_file = 'work_dirs/cascade_rcnn_x101_64x4d_fpn_1x/epoch_12.pth'

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    print('=>using config file: {}'.format(config_file))
    print('=>loadig model: {} success!'.format(checkpoint_file))


    # test a list of images and write the results to image files
    sampels_dir = 'data/ice_data/samples/'
    files = os.listdir(sampels_dir)
    imgs = [os.path.join(sampels_dir, file) for file in files if file[-4:]=='.jpg']

    start = time.time()
    results = []
    for i in range(len(imgs)):
        result = inference_detector(model, imgs[i])
        obj = {}
        obj['img_name'] = imgs[i]
        obj['bbox_dt'] = result
        results.append(obj)
        # show_result(imgs[i], result, [model.CLASSES], out_file='outputs/result_samples_{}.jpg'.format(i))

        #
    mmcv.dump(results, 'outputs/samples_results.json')
    t = time.time()-start
    print(t, len(imgs), t/len(imgs))
    print()
    print('inference finished ...')

if __name__=='__main__':
    main()