import numpy as np
import random
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from transforms import (
    VideoToTensor,
    VideoNormalize,
    VideoRandomVerticalFlip,
    VideoRandomHorizontalFlip,
)
from problem import (
    get_train_data,
    get_test_data,
    WeightedClassificationError,
    AreaUnderCurveError,
)
from models import simple_model, pretrained_models, convLSTM
from losses import WeightedCrossEntropy, WeightedCrossError
from utils import framerate, load_config

from dataset import VideoDataset, ImageDataset
from submissions.submission.turku_v1 import VideoClassifier


# For reproducing results i.e debbuging
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_weighted_sampler(data_path="../"):
    """Oversampling sampler because the dataset is unbalanced"""
    videos_train, label_train = get_train_data(data_path)
    labs, counts = np.unique(label_train, return_counts=True)
    weight = 1.0 / counts

    ordenize_token = np.vectorize(ord)
    target = ordenize_token(label_train) - ord("A")

    samples_weight = weight[target]
    samples_weight = torch.from_numpy(samples_weight)

    sampler = WeightedRandomSampler(
        samples_weight, len(samples_weight), replacement=True
    )
    return sampler


def get_data(data_path):
    videos_train, label_train = get_train_data(data_path)
    videos_test, labels_test = get_test_data(data_path)

    videos = videos_train + videos_test
    labels = np.concatenate([label_train, labels_test])

    return videos, labels


def get_image_loader(transforms, batch_size, closest_frame=-1, data_path="../"):
    """
    Create image train, validation and test dataloader from the project path
    A single transforms for every loader because there is no data augmentation
    """

    videos_train, label_train = get_train_data(data_path)
    videos_test, labels_test = get_test_data(data_path)

    split = len(videos_test) // 2
    videos_validation, videos_test = videos_test[split:], videos_test[:split]
    label_validation, label_test = labels_test[split:], labels_test[:split]

    train_dataset = ImageDataset(
        videos_train, label_train, closest_frame=closest_frame, transform=transforms
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    validation_dataset = ImageDataset(
        videos_validation,
        label_validation,
        closest_frame=closest_frame,
        transform=transforms,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

    test_dataset = ImageDataset(
        videos_test, label_test, closest_frame=closest_frame, transform=transforms
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, validation_loader, test_loader


def get_videos_loader(transform_train, transform_test, batch_size, data_path="../"):
    """
    Create video train, validation and test dataloader from the project path
    """

    videos_train, label_train = get_train_data(data_path)
    videos_test, label_test = get_test_data(data_path)

    split = len(videos_test) // 2
    videos_validation, videos_test = videos_test[split:], videos_test[:split]
    label_validation, label_test = label_test[split:], label_test[:split]

    # sampler = get_sampler()
    train_dataset = VideoDataset(videos_train, label_train, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    validation_dataset = VideoDataset(
        videos_validation, label_validation, transform=transform_test
    )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

    test_dataset = VideoDataset(videos_test, label_test, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, validation_loader, test_loader


def get_model_and_transform(config, dict_model=pretrained_models.dict_models):
    if config.video:
        cnn_layers = config.cnn_layers
        input_size = cnn_layers[-1] * 2
        channel_first = False
        sub_factor, sequence_length = framerate(config.pred_time)

        if config.model_name == "convLSTM":
            net = convLSTM.CellEncoder(
                sequence_length=sequence_length, features=cnn_layers
            )

        elif config.model_name == "simple_model":
            net = simple_model.SimpleNetwork(
                cnn_layers, input_size, sequence_length, config.device
            )

        # I decided not to add data augmentation here because it complicates the code a lot
        # with the addition of the k fold for little improvement

        transform = torchvision.transforms.Compose(
            [
                VideoToTensor(
                    end_vid=config.pred_time,
                    channel_first=channel_first,
                    sub_factor=sub_factor,
                    extraction_before_pred=sequence_length,
                ),
                VideoNormalize(),
            ]
        )

        return net, transform

    if config.image:
        backbone = config.backbone_name

        net = dict_model[backbone]["name"]
        weights = dict_model[backbone]["weights"]
        weights = weights.DEFAULT
        preprocess_transforms = weights.transforms()

        model = net(weights=weights)

        for param in model.parameters():
            param.requires_grad = False

        last_layer = dict_model[backbone]["last_layer_name"]
        last_layer_dim = dict_model[backbone]["last_layer_dim"]

        head = nn.Sequential(
            nn.Linear(last_layer_dim, config.intermediate_layers_size),
            nn.ReLU(),
            nn.Linear(config.intermediate_layers_size, config.features_dim),
        )
        setattr(model, last_layer, head)

        backbone_transform = transforms.Compose(
            [transforms.ToTensor(), preprocess_transforms]
        )

        return model, backbone_transform


def train(config):
    videos, labels = get_data(config.data_path)

    _, transform = get_model_and_transform(config)

    criterion = WeightedCrossEntropy(alpha=config.alpha, device=config.device)
    WCError = WeightedCrossError(device=config.device)

    kfolds = KFold(n_splits=config.k_folds, shuffle=True)

    if config.image:
        dataset = ImageDataset(
            videos, labels, closest_frame=config.closest_frame, transform=transform
        )

    if config.video:
        dataset = VideoDataset(videos, labels, transform=transform)

    results = {}

    for fold, (train_id, test_id) in enumerate(kfolds.split(dataset)):
        # writer = SummaryWriter(
        #     f"runs/..._research/fold {fold} ... {...}"
        # )
        print(f"FOLD {fold}")

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_id)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_id)

        train_loader = DataLoader(
            dataset, batch_size=config.batch_size, sampler=train_subsampler
        )
        test_loader = DataLoader(
            dataset, batch_size=config.batch_size, sampler=test_subsampler
        )

        model, _ = get_model_and_transform(config)

        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        scaler = torch.cuda.amp.GradScaler()  # to have float16 training
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, verbose=True
        )

        model.to(device=config.device)

        step = 0
        for epoch in range(config.nb_epochs):
            model.train()
            train_loss = 0.0
            train_WCE = 0.0
            val_loss = 0.0
            val_WCE = 0.0

            loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
            for batch_idx, (data, targets) in loop:
                data = data.to(device=config.device)
                targets = targets.to(device=config.device)

                with torch.cuda.amp.autocast():
                    pred = model(data)
                    loss = criterion(input=pred, target=targets)
                    wce_error = WCError(input=pred, target=targets)

                # back-propagation
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                train_WCE += wce_error.item()

                # Clip grad for LSTM
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

                # Update progress bar
                loop.set_description(f"Epoch[{epoch}/{config.nb_epochs}]")
                loop.set_postfix(loss=loss.item())

            train_loss = train_loss / len(train_loader)
            train_WCE = train_WCE / len(train_loader)

            # writer.add_scalar("Training Loss", train_loss, global_step=step)
            # writer.add_scalar("Training Error", train_WCE, global_step=step)

            print(
                f"{epoch} / {config.nb_epochs} - train_loss : {train_loss:.{4}} "
                f"- train_WCE : {train_WCE:.{4}}"
            )

            # -------- Evaluation of the model with validation set --------
            model.eval()
            with torch.no_grad():
                # ------------ Compute scores ------------
                for imgs_val, labels_val in test_loader:
                    imgs_val = imgs_val.to(device=config.device)
                    labels_val = labels_val.to(device=config.device)

                    pred = model(imgs_val)
                    current_val_loss = criterion(pred, labels_val)
                    current_val_WCE = WCError(pred, labels_val)
                    val_loss += current_val_loss.item()
                    val_WCE += current_val_WCE.item()

                val_loss = val_loss / len(test_loader)
                val_WCE = val_WCE / len(test_loader)

                # writer.add_scalar("Test Loss", val_loss, global_step=step)
                # writer.add_scalar("Test Error", val_WCE, global_step=step)

                print(
                    f"- val_loss   : {val_loss:.{4}} " f"- val_WCE   : {val_WCE:.{4}}"
                )

            step += 1

            results[fold] = val_WCE

            scheduler.step(train_loss)

    # Print fold results
    print(f"K-FOLD CROSS VALIDATION RESULTS FOR {config.k_folds} FOLDS")
    print("--------------------------------")
    sum = 0.0
    for key, value in results.items():
        print(f"Fold {key}: {value} %")
        sum += value
    print(f"Average: {sum / len(results.items())} %")


def submission_test(
    pred_times=(27, 32, 37, 40, 44, 48, 53, 58, 63, 94), data_path="../"
):
    videos_train, label_train = get_train_data(data_path)
    videos_test, label_test = get_test_data(data_path)

    # we need to convert labels (str) to 1-hot encoding (n, 8)
    labels_test = label_test.reshape(-1, 1)
    labels_train = label_train.reshape(-1, 1)

    enc = OneHotEncoder()
    enc.fit(labels_test)
    y_train = enc.transform(labels_train)
    y_true = enc.transform(labels_test)

    y_train = y_train.toarray()
    y_true = y_true.toarray()  # add .toarray() to fix an error below

    err = np.zeros((len(pred_times),))
    all_preds = []

    for time_idx, my_pred_time in enumerate(pred_times):
        wce = WeightedClassificationError(time_idx=time_idx)

        my_model = VideoClassifier()
        my_model.fit(videos_train, label_train, pred_time=my_pred_time)

        y_train_pred = my_model.predict(videos_train, pred_time=my_pred_time)
        err_train = wce.compute(y_true=y_train, y_pred=y_train_pred)
        print("at time", my_pred_time, " my training error is:", err_train)

        y_pred = my_model.predict(videos_test, pred_time=my_pred_time)
        all_preds.append(y_pred)
        err[time_idx] = wce.compute(y_true=y_true, y_pred=y_pred)
        print(wce.name, " at time", my_pred_time, "is:", err[time_idx])

    all_preds = np.concatenate(all_preds, axis=1)
    auc = AreaUnderCurveError(
        score_func_name="classification", prediction_times=pred_times
    )
    final_score = auc.compute(y_true=y_true, y_pred=all_preds)
    print(f"The final score of the model is {final_score}")


if __name__ == "__main__":
    # If we finetuned models lots of warnings might be raised from older models implementation
    import warnings

    warnings.filterwarnings("ignore")

    config_path = "./config.yaml"
    config = load_config(config_path)

    train(config)
    # submission_test()
