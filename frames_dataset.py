"""
Copyright Snap Inc. 2023. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import os
from collections import defaultdict
from functools import partial

import numpy as np
import torch
from skimage import io
from skimage.transform import resize
from torch.utils.data import Dataset
from tqdm import tqdm


class FramesDataset(Dataset):
    """
    Dataset of videos. Each video should be a folder with frames. Each frame should be an image.
    The structure should be this:
       dataset_name:
         train:
            video_1:
              001.png
              002.png
              ...
            video_2:
              001.png
              002.png
              ...
            ...
        test:
            video_1:
              001.png
              002.png
              ...
    Note that regular soring is used to determine order of frames, so pad the names with zeros.
    """

    def __init__(self,
                 root_dir,
                 resolution=(256, 256),
                 is_train=True,
                 max_frames=None,
                 max_videos=None,
                 repeat=1,
                 visualization_videos=4,
                 visualization_frames=None):
        """
        :param root_dir: the main directory of the dataset with train and test sub-folders
        :param resolution: resolution of the images, each image will be resized to this resolution
        :param is_train: load train or test subset
        :param max_frames: if not None, use this number of random frames from each video
        :param max_videos: if not None, use only this number of random videos
        :param repeat: if the size of the dataset is too small for one epoch, we can repeat dataset several times

        :param visualization_videos: either: -list with the name of videos to visualize or
                                             -int specifying number of random videos
        :param visualization_frames: some videos may be long, specify the number of frames to visualize
        """

        self.root_dir = root_dir
        folder = 'train' if is_train else 'test'
        folder = os.path.join(root_dir, folder)

        # Select which videos will be used
        videos = list(sorted(os.listdir(folder)))
        if max_videos is not None:
            np.random.seed(0)
            videos = sorted(np.random.choice(videos, max_videos, replace=False))

        # Select videos for visualization
        self.visualization_videos = []
        if isinstance(visualization_videos, int):
            np.random.seed(0)
            visualization_videos = min(len(videos), visualization_videos)
            visualization_videos = np.random.choice(videos, visualization_videos, replace=False)

        # Create a list of all frames (self.frames) and mapping from video_id to frames
        np.random.seed(0)
        self.frames = []
        self.id_to_frames = defaultdict(list)
        for video_id, video_name in tqdm(enumerate(videos)):
            path = os.path.join(folder, video_name)
            video_frames = []
            frames = sorted(os.listdir(path))

            if max_frames is not None and len(frames) > max_frames:
                frames = np.random.choice(frames, max_frames, replace=False)
            frames = sorted(frames)

            for frame_index, frame_name in enumerate(frames):
                info = {'video_id': video_id, 'frame_index': frame_index, 'path': os.path.join(path, frame_name)}

                self.frames.append(info)
                video_frames.append(info)
                self.id_to_frames[video_id].append(info)

            if video_name in visualization_videos:
                if visualization_frames is None:
                    self.visualization_videos.append(video_frames)
                else:
                    self.visualization_videos.append(video_frames[:visualization_frames])

        print("Found ", len(self.id_to_frames), " videos.")
        print("Found ", len(self.frames), "frames.")

        self.is_train = is_train
        self.resize_shape = resolution
        self.resize_fn = partial(resize, output_shape=tuple(self.resize_shape))
        self.max_videos = len(videos)
        self.ids_list = list(self.id_to_frames.keys()) * repeat

    def get_visualization_videos(self):
        """
        Read all the frames from visualization videos and pack them into a Tensors.
        :return: list of videos, each video is a dict:
                                                 video['frame'] [n_frames, 3, h, w]
                                                 video['video_id'] [n_frames,]
        """
        out_videos = []

        for frames_data in self.visualization_videos:
            video = defaultdict(list)
            for frame_info in frames_data:
                frame_dict = self.get_item_from_path(frame_info['video_id'], frame_info['path'])
                video['frame'].append(np.array(frame_dict['frame'])[np.newaxis])
                video['video_id'].append(np.array(frame_dict['video_id'])[np.newaxis])

            frames = np.concatenate(video['frame'], axis=0)
            frames = torch.Tensor(frames)
            video_id = np.concatenate(video['video_id'], axis=0)
            video_id = torch.LongTensor(video_id)
            video = {'frame': frames, 'video_id': video_id}

            out_videos.append(video)
        return out_videos

    def __len__(self):
        return len(self.ids_list)

    def get_item_from_path(self, video_id, path):
        """
        :return" a dict with 2 field:
                out['frame']: [3, h, w]
                out['video_id']: int
        """
        image = self.resize_fn(io.imread(path))
        image = np.array(image, dtype='float32').transpose((2, 0, 1))
        return {'frame': image, 'video_id': video_id}

    def __getitem__(self, idx):
        """
        :return" a dict with 2 field:
                out['frame']: [3, h, w]
                out['video_id']: int
        """
        video_id = self.ids_list[idx]
        frame_list = self.id_to_frames[video_id]
        frame_info = frame_list[np.random.randint(0, len(frame_list))]
        return self.get_item_from_path(frame_info['video_id'], frame_info['path'])
