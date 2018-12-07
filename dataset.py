import datetime
import time

import cv2
import numpy as np
import pandas as pd
import torch
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize

from utils import log


class TrainData:
    def __init__(self, data_dir):
        start_time = time.time()

        df = pd.read_csv(
            "{}/train.csv".format(data_dir),
            index_col="Id",
            converters={"Target": lambda target: tuple(map(int, str(target).split(" ")))}
        )

        train_set_ids, val_set_ids = train_test_split(df.index, test_size=0.2, random_state=42)

        self.df = df
        self.train_set_df = df[df.index.isin(train_set_ids)].copy()
        self.val_set_df = df[df.index.isin(val_set_ids)].copy()

        end_time = time.time()
        log("Time to prepare train data: {}".format(str(datetime.timedelta(seconds=end_time - start_time))))


class TrainDataset(Dataset):
    def __init__(self, df, data_dir, num_categories, image_size, augment):
        super().__init__()
        self.df = df
        self.data_dir = data_dir
        self.num_categories = num_categories
        self.image_size = image_size
        self.augment = augment
        self.augmentor = iaa.Sequential([
            iaa.Sometimes(
                0.5,
                iaa.OneOf([
                    iaa.Affine(rotate=90),
                    iaa.Affine(rotate=180),
                    iaa.Affine(rotate=270),
                    iaa.Fliplr(0.5),
                    iaa.Flipud(0.5)
                ])),
            iaa.Sometimes(0.5, iaa.Affine(shear=(-16, 16))),
            iaa.Sometimes(0.5, iaa.Multiply((0.8, 1.2), per_channel=True))
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        id = self.df.index[index]
        categories = self.df.iloc[index].Target

        image = load_image(self.data_dir + "/train", id, self.image_size)

        if self.augment:
            image = self.augmentor.augment_image(image)

        image_t = image_to_tensor(image)
        categories_t = categories_to_tensor(categories, self.num_categories)

        assert categories_t.sum() > 0, "image has no targets: {} -> {}".format(id, categories)

        image_t = normalize(
            image_t,
            mean=(0.0804419, 0.0526298, 0.0547470, 0.0827089),
            std=(0.1496247, 0.1122266, 0.1560370, 0.1496669)
        )

        return image_t, categories_t


class TestData:
    def __init__(self, data_dir):
        start_time = time.time()

        self.test_set_df = pd.read_csv("{}/sample_submission.csv".format(data_dir), index_col="Id", usecols=["Id"])

        end_time = time.time()
        log("Time to prepare test data: {}".format(str(datetime.timedelta(seconds=end_time - start_time))))


class TestDataset(Dataset):
    def __init__(self, df, data_dir, image_size):
        super().__init__()
        self.df = df
        self.data_dir = data_dir
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        id = self.df.index[index]

        image = load_image(self.data_dir + "/test", id, self.image_size)

        image_t = image_to_tensor(image)

        image_t = normalize(
            image_t,
            mean=(0.0804419, 0.0526298, 0.0547470, 0.0827089),
            std=(0.1496247, 0.1122266, 0.1560370, 0.1496669)
        )

        return (image_t,)


def load_image(base_dir, id, image_size):
    r = load_image_channel("{}/{}_red.png".format(base_dir, id), image_size)
    g = load_image_channel("{}/{}_green.png".format(base_dir, id), image_size)
    b = load_image_channel("{}/{}_blue.png".format(base_dir, id), image_size)
    y = load_image_channel("{}/{}_yellow.png".format(base_dir, id), image_size)
    return np.stack([r, g, b, y], axis=0)


def load_image_channel(file_path, image_size):
    channel = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if channel.shape[0] != image_size:
        channel = cv2.resize(channel, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return channel


def image_to_tensor(image):
    return torch.from_numpy(image / 255.).float()


def categories_to_tensor(categories, num_categories):
    return torch.tensor([1 if i in categories else 0 for i in range(num_categories)]).float()
