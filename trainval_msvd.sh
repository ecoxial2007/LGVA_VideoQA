python3 src/trainval.py \
        --dataset 'msvd_oe' \
        --data_path './data/Annotation/msvd' \
        --feature_path '/home/liangx/Data/MSVD'\
        --batch_size 64

#
python3 src/test.py \
        --dataset 'msvd_oe' \
        --data_path './data/Annotation/msvd' \
        --feature_path '/home/liangx/Data/MSVD'\
        --checkpoint './checkpoints/msvd_oe/ckpt_0.5351519584655762.pth' \
        --batch_size 64
