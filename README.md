# Language-Guided Visual Aggregation for Video Question Answering
This is the implementation of our paper, all features and weights will be released on github. 
You can also extract video and text features yourself according to our code and documentation.

## Environment
This code is tested with:
- Ubuntu 20.04
- PyTorch >= 1.8
- CUDA >= 10.1

```
# create your virtual environment
conda create --name lgva python=3.7
conda activate lgva

# dependencies
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.1 -c pytorch
conda install pandas

# optional (for feature extraction); see also tools/*.py
pip install git+https://github.com/openai/CLIP.git
```

## Dataset
- Annotation: check `./data/Annotation`
- Source data:
  - NExT-QA: https://xdshang.github.io/docs/vidor.html
  - MSR-VTT & MSVD: https://github.com/xudejing/video-question-answering
  - ActivityNet-QA: https://github.com/MILVLG/activitynet-qa
  - TGIF: https://github.com/YunseokJANG/tgif-qa

## Feature extraction
Please refer to `./tools/extract_embedding.py`

## Pre-extracted Features

| dataset    | frame | bbox | caption | question&answer |
|------------|-------|------|---------|-----------------|
| NExT-QA |	[BaiduDisk](https://pan.baidu.com/s/1bohs6bSH2tLyOXDqvOTxQw?pwd=6666) |	uploading |	uploading |	[BaiduDisk](https://pan.baidu.com/s/11k6ZuVE6vmWtkOX9g9xMaA?pwd=6666)	|
| MSVD    |	uploading |	uploading |	[BaiduDisk](https://pan.baidu.com/s/1oYGIiFIwyZWOeL1RJvb6-g?pwd=6666) |	[BaiduDisk](https://pan.baidu.com/s/14FCF7DKb_fRiGx78s3rFIA?pwd=6666)	|
| MSRVTT  |	uploading |	uploading |	uploading |	uploading |	uploading |

Due to the large number of videos in TGIF and ActivityNet, we do not plan to upload the features. You can process the original videos using a simple feature extraction script. Similarly, extracting text features (such as questions and answers) does not take much time, and you can extract them on your own based on the json files.

## Train & Val & Test
Check `trainval_msvd.sh` & `trainval_nextqa.sh`
```
python3 src/trainval.py \
        --dataset 'nextqa_mc' \
        --data_path './data/Annotation' \
        --feature_path '/home/liangx/Data/NeXt-QA'\
        --batch_size 256

python3 src/test.py \
        --dataset 'nextqa_mc' \
        --data_path './data/Annotation' \
        --feature_path '/home/liangx/Data/NeXt-QA'\
        --checkpoint './checkpoints/nextqa_mc/ckpt_0.6112890243530273.pth' \
        --batch_size 256 \
        --visible
```

### LICENSE / Contact

We release this repo under the open [MIT License](LICENSE). 

## Citations
```
@article{Liang2023LanguageGuidedVA,
  title={Language-Guided Visual Aggregation Network for Video Question Answering},
  author={Xiao Liang and Di Wang and Quan Wang and Bo Wan and Lingling An and Lihuo He},
  journal={Proceedings of the 31st ACM International Conference on Multimedia},
  year={2023},
  url={https://api.semanticscholar.org/CorpusID:264492577}
}
```

# Acknowledgements
We reference the excellent repos of NeXT-QA, VGT, ATP, CLIP, in addition to other specific repos to the datasets/baselines we examined (see paper). If you build on this work, please be sure to cite these works/repos as well.
