import numpy as np
from torch.utils.data import Dataset

from utils import brightness_correction, cell_detection


class VideoDataset(Dataset):
    def __init__(self, videos, y, transform=None):
        self.transform = transform
        self.videos = videos
        self.labels = y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        video = self.videos[idx]
        label = self.labels[idx]

        # To use Pytorch CrossEntropy
        ordenize_token = np.vectorize(ord)
        label = ordenize_token(label) - ord("A")

        if self.transform:
            video = self.transform(video)

        return video, label


class ImageDataset(Dataset):
    def __init__(
        self, videos, y, end_time=94, start_time=23.75, closest_frame=-1, transform=None
    ):
        self.transform = transform
        self.videos = videos
        self.labels = y

        # We only keep the k-th last frame before pred_time to accelerate training and reduce the influence of the first
        # frames that might not be relevant where predicting time is large
        if closest_frame != -1:
            start_time = max(end_time - closest_frame * 0.25, start_time)

        # start_time is fixed at 23.75 because it is the latest start of the dataset
        # It is easier to deal with a dataset of same duration videos
        # end_time can be reduced depending on our inference time
        self.start_time = start_time
        self.len_video = int((end_time - start_time) * 4) + 1
        self.len_dataset = len(videos) * self.len_video

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        idx_vid = idx // self.len_video
        idx_time = idx % self.len_video
        timestamp = idx_time / 4 + self.start_time

        video = self.videos[idx_vid]
        frame = video.read_frame(frame_time=timestamp)
        frame = brightness_correction(frame)
        frame = cell_detection(frame)

        label = self.labels[idx_vid]

        # Pour utiliser la CrossEntropy de Pytorch
        ordenize_token = np.vectorize(ord)
        label = ordenize_token(label) - ord("A")

        if self.transform:
            frame = np.repeat(frame[..., np.newaxis], 3, -1)  # (H, W) -> (H, W, 3)
            frame = self.transform(frame)

        return frame, label
