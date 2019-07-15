# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/root/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/root/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/root/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/root/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda deactivate
conda activate mmlab

########################################## step 1 data #################################################
# data prepare
cd /root/wangxin/test
# 1  create csv
python total_pnm.py -c 30 -r /dataset/training

# 2 csv 2 coco  
#out: /root/wangxin/test/coco/annotations/instances_train2017.json
python csv2coco.py

#3 csv -> images
# -o default='/data/imgs'
python multi.py


########################################## step 2 detection #################################################
cd /root/zhongwei/ice_mmdetection
python tools/inference_batch.py configs/cascade_rcnn_x101_64x4d_fpn_1x.py ../mm_models/epoch_12.pth --out /root/chenxinli/results.pkl


########################################## step 3 classification #################################################
cd /root/chenxinli/
python readcoco.py 
python inference.py --c 30



