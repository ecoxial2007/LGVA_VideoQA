import json

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import cv2
import glob
import os
import requests
import skvideo.io
import json
from io import BytesIO
from PIL import Image
import numpy as np
pylab.rcParams['figure.figsize'] = 20, 12
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
vg_cate_dict = {}

with open('synsets.json', 'r') as f:
    vg_cate_list = json.load(f)

for vg_cate in vg_cate_list:
    synset_name = vg_cate['synset_name']
    word_cate = synset_name[-4:]
    word = synset_name[:-5]
    if word_cate not in vg_cate_dict.keys():
        vg_cate_dict[word_cate] = [word]
    else:
        vg_cate_dict[word_cate].append(word)

coco_cate_dict = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck',
           9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
           16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra',
           25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
           35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
           41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup',
           48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
           56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
           64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
           75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
           82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
           90: 'toothbrush'}

def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(response).convert("RGB")
    # pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img, caption):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)


if __name__ == '__main__':
    config_file = "../configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
    weight_file = "../MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth"

    # config_file = "configs/pretrain/glip_Swin_L.yaml"
    # weight_file = "MODEL/glip_large_model.pth"

    cfg.local_rank = 0
    cfg.num_gpus = 1
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
    coco_caption = '. '.join(coco_cate_dict.values())
    # vg_caption = []
    # for word_type in vg_cate_dict.keys():
    #     if 'n.' in word_type:
    #         vg_caption.extend(vg_cate_dict[word_type])
    # vg_caption = '. '.join(vg_caption)

    glip_demo = GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.0,
        show_mask_heatmaps=False
    )
    caption = coco_caption#vg_caption#
    video_root = '/home/liangx/Data/NExT-QA/'
    bbox_path = './nextqa-glip-bbox.json'
    bbox_dict = {}
    video_paths = glob.glob(os.path.join(video_root, '*', '*', '*', '*.mp4'))


    num_clips = 16
    frame_per_clip = 4
    for video_path in video_paths:
        print(video_path)
        video_data = skvideo.io.vread(video_path)
        _, name = os.path.split(video_path)

        metadata = skvideo.io.ffprobe(video_path)
        total_frame = video_data.shape[0]
        sample_rate = total_frame / num_clips / frame_per_clip
        bbox_dict[name] = []
        for mark_id in range(num_clips*frame_per_clip):

            fid = int(mark_id*sample_rate)
            frame = video_data[fid]
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            result, predictions, entity_names = glip_demo.run_on_web_image(image, caption, 0.0)
            entity_name = predictions.get_field("labels").tolist()

            if len(predictions) < 10:
                continue


            bboxes_per_frame = []
            for bid in range(10):
                x1, y1, x2, y2 = map(int, predictions.bbox[bid])
                entity_name = entity_names[bid]
                bbox_per_frame = {
                    'bbox': [x1, y1, x2, y2],
                    'label': entity_name
                }
                bboxes_per_frame.append(bbox_per_frame)
                result = cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 255, 0), thickness=2)
            # print(entity_names)
            bbox_dict[name].append(bboxes_per_frame)
            # cv2.imshow('a',frame)
            # cv2.waitKey()


        with open(bbox_path, 'w') as fj:
            fj.write(json.dumps(bbox_dict))
