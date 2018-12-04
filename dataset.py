import datetime
import time

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from utils import log


class TrainData:
    def __init__(self, data_dir):
        start_time = time.time()

        df = pd.read_csv(
            "{}/train.csv".format(data_dir),
            index_col="Id",
            converters={"Target": lambda target: map(int, target.split(" "))}
        )

        train_set_ids, val_set_ids = train_test_split(df.index, test_size=0.2, random_state=42)

        self.train_set_df = df[df.index.isin(train_set_ids)].copy()
        self.val_set_df = df[df.index.isin(val_set_ids)].copy()

        end_time = time.time()
        log("Time to prepare train data: {}".format(str(datetime.timedelta(seconds=end_time - start_time))))


class TrainDataset(Dataset):
    def __init__(self, df, data_dir, num_categories):
        super().__init__()
        self.df = df
        self.data_dir = data_dir
        self.num_categories = num_categories

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        id = self.df.index[index]
        categories = self.df.Target[index]

        image = load_image(self.data_dir + "/train", id)

        image = image_to_tensor(image)
        categories = categories_to_tensor(categories, self.num_categories)

        # image = normalize(image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        return image, categories


def load_image(base_dir, id):
    r = cv2.imread("{}/{}_red.png".format(base_dir, id), 0)
    g = cv2.imread("{}/{}_green.png".format(base_dir, id), 0)
    b = cv2.imread("{}/{}_blue.png".format(base_dir, id), 0)
    y = cv2.imread("{}/{}_yellow.png".format(base_dir, id), 0)
    return np.stack([r, g, b, y], axis=0)


def image_to_tensor(image):
    return torch.from_numpy(image / 255.).float()


def categories_to_tensor(categories, num_categories):
    return torch.tensor([1 if i in categories else 0 for i in range(num_categories)]).float()
