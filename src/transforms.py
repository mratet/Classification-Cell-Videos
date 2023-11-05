# github source : https://github.com/YuxinZhaozyx/pytorch-VideoDataset/blob/master/transforms.py

import numpy as np
import torch
import torchvision
import random

from utils import brightness_correction, cell_detection


class VideoToTensor(object):
    def __init__(self, start_vid=22, end_vid=94, extraction_before_pred=-1, height=160, width=160, sub_factor=1, in_channels=1, channel_first=False, padding=True):

        self.start_vid = start_vid
        self.end_vid = end_vid  # last frame used
        self.sub_factor = sub_factor
        self.padding = padding
        self.extraction = extraction_before_pred

        self.channels = in_channels
        self.channel_first = channel_first

        self.height = height
        self.width = width

    def __call__(self, video):

        frame_times = video.frame_times
        frame_times = frame_times[frame_times <= self.end_vid]

        if self.extraction != -1:
            frame_times = frame_times[-self.extraction:]
            self.sub_factor = 1
        else:
            frame_times = frame_times[self.start_vid <= frame_times]

            if self.padding:
                start_time = min(frame_times)
                if start_time > self.start_vid:
                    nb_frame_to_add = int((start_time - self.start_vid) * 4)
                    fill = start_time * np.ones(nb_frame_to_add, dtype="float64")
                    frame_times = np.concatenate((fill, frame_times))

        time_len = len(frame_times) // self.sub_factor

        if self.channel_first:
            frames = torch.FloatTensor(self.channels, time_len, self.height, self.width)
        else:
            frames = torch.FloatTensor(time_len, self.channels, self.height, self.width)

        p = np.random.randint(low=0, high=self.sub_factor, size=1)[0]

        for idx in range(time_len):
            frame_time = frame_times[p + self.sub_factor * idx]
            frame = video.read_frame(frame_time=frame_time)
            frame = brightness_correction(frame)
            frame = cell_detection(frame, height=self.height, width=self.width)
            frame = torch.from_numpy(frame)

            if self.channel_first:
                frames[0, idx, :, :] = frame.float()
            else:
                frame = frame.unsqueeze(0)
                frames[idx, :, :, :] = frame.float()

        frames /= 255
        return frames


class VideoRandomCrop(object):

    def __init__(self, size):
        assert len(size) == 2
        self.size = size

    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video (L x C x H x W) to be cropped.
        Returns:
            torch.Tensor: Cropped video (L x C x h x w).
        """

        H, W = video.size()[2:]
        h, w = self.size
        assert H >= h and W >= w

        top = np.random.randint(0, H - h)
        left = np.random.randint(0, W - w)

        video = video[:, :, top:top + h, left:left + w]

        return video


class VideoRandomVerticalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video):
        if random.random() < self.p:
            video = video.flip([2])

        return video


class VideoRandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video):
        if random.random() < self.p:
            # horizontal flip the video
            video = video.flip([3])

        return video


class VideoNormalize(object):

    def __init__(self, mean=0.472, std=0.2828):
        self.mean = mean
        self.std = std

    def __call__(self, video):

        video = (video - self.mean) / self.std

        return video


class VideoResize(object):

    def __init__(self, size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, video):

        h, w = self.size
        L, C, H, W = video.size()
        rescaled_video = torch.FloatTensor(L, C, h, w)

        # use torchvision implementation to resize video frames
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(self.size, self.interpolation),
            torchvision.transforms.ToTensor(),
        ])

        for l in range(L):
            frame = video[l, :, :, :]
            frame = transform(frame)
            rescaled_video[l, :, :, :] = frame

        return rescaled_video

