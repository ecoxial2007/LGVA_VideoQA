#python3 src/trainval.py \
#        --dataset 'nextqa_mc' \
#        --data_path './data/Annotation' \
#        --feature_path '/home/liangx/Data/NeXt-QA'\
#        --batch_size 256


python3 src/test.py \
        --dataset 'nextqa_mc' \
        --data_path './data/Annotation/nextqa_visible' \
        --feature_path './data/video/NExT-QA'\
        --checkpoint './checkpoints/nextqa_mc/ckpt_0.6112890243530273.pth' \
        --batch_size 4 \
        --visible

