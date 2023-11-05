import torch
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import yaml
import munch


def load_config(config_path: str):
    try:
        with open(config_path) as fp:
            config_dict = yaml.safe_load(fp)
            config_dict = munch.munchify(config_dict)
    except yaml.YAMLError as err:
        raise RuntimeError(err)

    return config_dict


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


def get_mean_std(loader):
    """was used to computed mean/std of the dataset from the VideoLoader

    train_loader, _ = get_image_loader(transforms=transforms.ToTensor(), batch_size=64)
    mean, std = get_mean_std(train_loader)
    """
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5

    return mean, std


def framerate(pred_time, start_vid=22):
    """
    From start_vid and chose pred_time, return sub_factor
    that should be applied and the lenght of the sequence kept
    """

    pred_times = np.array([27, 32, 37, 40, 44, 48, 53, 58, 63, 94])
    sub_factors = np.array([2, 3, 5, 5, 6, 8, 10, 10, 13, 20])
    sequences_length = (4 * (pred_times - start_vid) + 1) // sub_factors

    assert pred_time in pred_times
    idx = np.where(pred_times == pred_time)
    sub_factor = sub_factors[idx][0]
    sequence_length = sequences_length[idx][0]
    return sub_factor, sequence_length
