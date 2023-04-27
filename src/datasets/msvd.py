"""
Simple video-language dataset class file, for illustration purposes. See comments below for relevant highlights.
"""
import json
import os
import random

import torch
from torch.utils import data
import h5py
import numpy as np


class VideoLanguageDataset(data.Dataset):
    """
    Example (simple) video-language dataset class, for illustration purposes for ATP training.
    """

    def __init__(self, args, split="train"):
        super().__init__()
        self.data_path = args.data_path
        self.feature_path = args.feature_path
        self.split = split
        self.n_frames = args.n_frames
        self.visible = args.visible

        with open(os.path.join(self.data_path, f'{split}_qa.json'), 'r') as jf:
            self.metadata = json.load(jf)

        self.text_features_question = h5py.File(os.path.join(self.feature_path, 'text_features_clip_all.h5'), 'r')
        self.text_features = h5py.File(os.path.join(self.feature_path, 'text_features_clip.h5'), 'r')
        self.text_cands_features = torch.tensor(self.text_features['all_candi_features'], dtype=torch.float32)
        self.text_caption_features = h5py.File(os.path.join(self.feature_path, 'text_features_blip_caption.h5'), 'r')['all_text_features']

        all_classes = self.text_features['all_classes']
        self.option = all_classes

        self.answer2label = {}
        for i, classes in enumerate(all_classes):
            self.answer2label[classes.decode('utf-8')] = i


        video_ids = []
        for f in self.metadata:
            video_ids.append(f['video_id'])
        self.video_ids = list(set(video_ids))
        print(self.split, len(self.metadata))
        print('num_classes', len(self.answer2label))


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        """
        Assuming torch files for each of the features for simplicity; adjust to fit your extracted features.
        (e.g. numpy, hdf5, json, etc.)
        """
        f = self.metadata[index]
        qid = int(f['id'])
        video_id = f['video_id']
        question = f['question']
        typee = question.split(' ')[0]
        answer = f['answer']
        labels_id = torch.tensor(self.answer2label[answer], dtype=torch.long)

        video_feature_path = os.path.join(self.feature_path, 'MSVD-clip-features', f'{video_id}.h5')
        video_features = h5py.File(video_feature_path, 'r')['video_features']
        video_features = torch.tensor(np.array(video_features), dtype=torch.float32)  # (L_video, D_in); L_video >> L

        bbox_feature_path = os.path.join(self.feature_path, 'MSVD-clip-bbox-features', f'{video_id}.h5')
        bbox_features = h5py.File(bbox_feature_path, 'r')['bbox_features']
        bbox_features = torch.tensor(np.array(bbox_features), dtype=torch.float32)  # (L_video, D_in); L_video >> L


        frame_idxs_gt = torch.randperm(len(video_features))[:self.n_frames]
        frame_idxs_gt = torch.sort(frame_idxs_gt).values

        video_features_sampled = video_features[frame_idxs_gt]  # (L, D_in)
        bbox_features_sampled = bbox_features[frame_idxs_gt]

        text_query_features = torch.tensor(self.text_features_question['question_features'][qid], dtype=torch.float32)
        text_query_features_global = torch.tensor(self.text_features['question_features'][qid], dtype=torch.float32)
        text_caps_features = torch.tensor(self.text_caption_features[qid], dtype=torch.float32)
        text_fqas_features = torch.tensor(self.text_frame_answer_features[qid], dtype=torch.float32)

        item_dict = {
            'video_features': video_features_sampled,
            'bbox_features': bbox_features_sampled,
            'text_caption_features': text_caps_features,
            'text_query_features': text_query_features_global,
            'text_query_token_features': text_query_features,
            'text_cands_features': self.text_cands_features,
            'labels_id': labels_id
        }

        if self.visible:
            item_dict.update({
                'additional_info': (video_id, question, 0, typee)
            })

        return item_dict