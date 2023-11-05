import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from torchvision.models import (
    efficientnet_v2_m,
    EfficientNet_V2_M_Weights,
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights,
    efficientnet_v2_l,
    EfficientNet_V2_L_Weights,
    vit_b_16,
    ViT_B_16_Weights,
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
)


class VideoClassifier(object):
    def __init__(self):
        self.nb_epochs = 50
        self.learning_rate = 0.005
        self.alpha = 0.1
        self.batch_size = 512
        self.device = "cuda"

        self.closest_frame = 48

        self.backbone_name = "mobilenet_v3_small"
        self.intermediate_layers_size = 200
        self.features_dim = 8

        self.dict_model = {
            "efficient_net_small": {
                "name": efficientnet_v2_s,
                "weights": EfficientNet_V2_S_Weights,
                "last_layer_name": "classifier",
                "last_layer_dim": 1280,
            },
            "efficient_net_medium": {
                "name": efficientnet_v2_m,
                "weights": EfficientNet_V2_M_Weights,
                "last_layer_name": "classifier",
                "last_layer_dim": 1280,
            },
            "efficient_net_large": {
                "name": efficientnet_v2_l,
                "weights": EfficientNet_V2_L_Weights,
                "last_layer_name": "classifier",
                "last_layer_dim": 1280,
            },
            "vision_transformer": {
                "name": vit_b_16,
                "weights": ViT_B_16_Weights,
                "last_layer_name": "heads",
                "last_layer_dim": 768,
            },
            "mobilenet_v3_small": {
                "name": mobilenet_v3_small,
                "weights": MobileNet_V3_Small_Weights,
                "last_layer_name": "classifier",
                "last_layer_dim": 576,
            },
            "mobilenet_v3_large": {
                "name": mobilenet_v3_large,
                "weights": MobileNet_V3_Large_Weights,
                "last_layer_name": "classifier",
                "last_layer_dim": 960,
            },
        }

        self.model, self.transform = self.get_model_and_transform()

    def get_model_and_transform(self):
        backbone = self.backbone_name

        net = self.dict_model[backbone]["name"]
        weights = self.dict_model[backbone]["weights"]
        weights = weights.DEFAULT
        preprocess_transforms = weights.transforms()

        model = net(weights=weights)

        for param in model.parameters():
            param.requires_grad = False

        last_layer = self.dict_model[backbone]["last_layer_name"]
        last_layer_dim = self.dict_model[backbone]["last_layer_dim"]

        head = nn.Sequential(
            nn.Linear(last_layer_dim, self.intermediate_layers_size),
            nn.ReLU(),
            nn.Linear(self.intermediate_layers_size, self.features_dim),
        )
        setattr(model, last_layer, head)

        backbone_transform = transforms.Compose(
            [transforms.ToTensor(), preprocess_transforms]
        )

        return model, backbone_transform

    def fit(self, videos: list, y, pred_time: float):
        dataset = ImageDataset(
            videos,
            y,
            end_time=pred_time,
            closest_frame=self.closest_frame,
            transform=self.transform,
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        criterion = WeightedCrossEntropy(alpha=self.alpha, device=self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scaler = torch.cuda.amp.GradScaler()  # to have float16 training

        self.model.to(device=self.device)
        self.model.train()

        for epoch in range(self.nb_epochs):
            for batch_idx, (data, targets) in enumerate(dataloader):
                data = data.to(device=self.device)
                targets = targets.to(device=self.device)

                with torch.cuda.amp.autocast():
                    pred = self.model(data)
                    loss = criterion(input=pred, target=targets)

                optimizer.zero_grad()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

    def predict(self, videos: list, pred_time: float):
        proba = torch.zeros([len(videos), self.features_dim])
        self.model.eval()

        with torch.no_grad():
            for i, video in enumerate(videos):
                frames = video.read_sequence(
                    begin_time=pred_time - 3.0, end_time=pred_time
                )
                pred = torch.zeros([1, 8]).to(device=self.device)

                for frame in frames:
                    # frame = video.read_frame(frame_time=pred_time)
                    frame = brightness_correction(frame)
                    frame = cell_detection(frame)
                    frame = np.repeat(
                        frame[..., np.newaxis], 3, -1
                    )  # (H, W) -> (H, W, 3)
                    frame = self.transform(frame).unsqueeze(0)
                    frame = frame.to(device=self.device)

                    pred += self.model(frame)
                proba[i] = F.softmax(pred.cpu(), dim=1)

        return proba


def brightness_correction(image):
    """
    Take in input a grey image and return a uniform brightness version
    """
    img = image.copy()
    cols, rows = image.shape
    brightness = np.sum(image) / (255 * cols * rows)
    minimum_brightness = 0.40
    ratio = brightness / minimum_brightness
    if ratio < 1:
        img = cv.convertScaleAbs(
            image, alpha=1 / ratio, beta=0
        )  # Brightness improvement
    return img


def cell_detection(img, width=160, height=160, display=False):
    """
    Take in input a grey image and return cropped image around the cell of size w * h
    The empirical best cropped size is 160 * 160
    """
    edges = cv.Canny(img, 100, 200)

    # to remove microscope wide circle
    border_size = 30
    n = len(edges)
    mask = np.zeros((n, n))
    mask[border_size:-border_size, border_size:-border_size] += 1
    edges = mask * edges

    M1 = cv.moments(edges)
    # calculate x,y coordinate of center
    cX = int(M1["m10"] / M1["m00"])
    cY = int(M1["m01"] / M1["m00"])

    w_crop, h_crop = 160, 160
    x1 = cX - w_crop // 2
    x2 = cX + w_crop // 2
    y1 = cY - h_crop // 2
    y2 = cY + h_crop // 2

    cropped_image = img[y1:y2, x1:x2]
    image = cv.resize(cropped_image, (width, height), interpolation=cv.INTER_LINEAR)

    if display:
        cv.circle(edges, (cX, cY), 5, (255, 255, 255), 10)

        cv.rectangle(img, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=3)

        plt.subplot(121), plt.imshow(img, cmap="gray")
        plt.title("Original Image"), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(edges, cmap="gray")
        plt.title("Edge Image"), plt.xticks([]), plt.yticks([])
        plt.show()

    return image


class ImageDataset(Dataset):
    def __init__(
        self,
        videos,
        y,
        end_time=94.0,
        start_time=23.75,
        closest_frame=-1,
        transform=None,
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


class WeightedCrossEntropy(nn.Module):
    def __init__(self, alpha, eps=1e-10, device="cuda"):
        super(WeightedCrossEntropy, self).__init__()

        self.alpha = alpha
        self.eps = eps

        weight_matrix = np.array(
            [
                [0, 1, 6, 10, 10, 10, 10, 10],
                [1, 0, 3, 10, 10, 10, 10, 10],
                [6, 3, 0, 2, 9, 10, 10, 10],
                [10, 10, 2, 0, 9, 9, 10, 10],
                [10, 10, 9, 9, 0, 8, 8, 8],
                [10, 10, 10, 9, 8, 0, 9, 8],
                [10, 10, 10, 10, 8, 9, 0, 9],
                [10, 10, 10, 10, 8, 8, 9, 0],
            ]
        )

        weight_matrix = weight_matrix / np.max(weight_matrix)
        self.weight_matrix = torch.tensor(weight_matrix).float().to(device=device)

    def forward(self, input, target):
        """
        Loss utilisé lors de l'entrainement des réseaux de neurones

        Input : y_pred, Vecteur (batch_size, 8) : prédiction du NN
        Attention Softmax ici donc pas besoin de l'inclure à la fin du réseau !

        Target : y_true, Vecteur (batch_size, 1) : label encodé incrémentalement
        """
        n_classes = 8

        # Compute the softmax/log-softmax of the predictions
        probs = F.softmax(input, dim=1)
        log_probs = F.log_softmax(input + self.eps, dim=1)

        # Create a one-hot encoding of the true labels
        target = F.one_hot(target, num_classes=n_classes).float()

        CrossEntropy = -torch.mean(torch.sum(target * log_probs, dim=1))

        WeightedCE = -torch.mean(
            torch.sum(
                torch.bmm(
                    torch.unsqueeze(torch.matmul(target, self.weight_matrix), 1),
                    torch.unsqueeze(torch.log(1 - probs + self.eps), 2),
                ),
                dim=1,
            )
        )
        # torch.sum --> Calcul de l'erreur par sample
        # torch.mean --> on prend la moyenne sur l'ensemble des échantillons

        loss = self.alpha * CrossEntropy + (1 - self.alpha) * WeightedCE

        return loss
