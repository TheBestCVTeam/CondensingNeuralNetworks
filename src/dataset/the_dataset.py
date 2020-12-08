from __future__ import division, print_function

import logging
import os
import warnings
from typing import List

import cv2
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from src.dataset.text_file_rec_conversion import (get_bb_fn, get_filename,
                                                  get_label)
from src.filter.base_filter import BaseFilter
from src.local.loc_folders import LocFolders
from src.utils.config import Conf
from src.utils.log import log
from src.utils.misc_func import ensure_folder_created
from src.utils.stopwatch import StopWatch
from src.utils.train_param import fil_to_str

warnings.filterwarnings("ignore")


def read_img_from_disk(path: str, should_crop: bool) -> Tensor:
    result = cv2.imread(path)

    # Crop Image
    if should_crop:
        result = perform_crop(result, path)

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # Convert to Tensor of correct size
    trans1 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    result = Image.fromarray(result)
    result = trans1(result)
    return result


def perform_crop(img: Tensor, path: str) -> Tensor:
    """
    Performs the crop of the image and returns the cropped image

    :param img: Image to be cropped
    :param path: Path to the image to be able to find the bounding box
    :return: Cropped image
    """

    # Initialize return variables
    result = img

    # Get the shape of input image
    real_h, real_w, c = result.shape

    try:
        with open(get_bb_fn(path), 'r') as f:
            material = f.readline()
            x, y, w, h, score = material.strip().split(' ')

        w = int(float(w))
        h = int(float(h))
        x = int(float(x))
        y = int(float(y))
        w = int(w * (real_w / 224))
        h = int(h * (real_h / 224))
        x = int(x * (real_w / 224))
        y = int(y * (real_h / 224))

        # Crop face based on its bounding box
        y1 = 0 if y < 0 else y
        x1 = 0 if x < 0 else x
        y2 = real_h if y1 + h > real_h else y + h
        x2 = real_w if x1 + w > real_w else x + w
        result = img[y1:y2, x1:x2, :]
    except Exception as e:
        log(f'Error cropping "{path}" with msg: {e}', logging.ERROR)
    return result


class TheDataset(Dataset):
    def __init__(self, im_txt: str, filters: List[BaseFilter.__class__],
                 *, should_save_img_after_filters: bool, max_size: int = None,
                 should_crop=True):
        super(TheDataset, self).__init__()
        imgs = []
        with open(im_txt) as f:
            imgs_list = f.readlines()

        # Truncate list length if max set and higher than max
        if max_size is not None:
            if len(imgs_list) > max_size:
                imgs_list = imgs_list[:max_size]

        for im_item in imgs_list:
            img_fn = get_filename(im_item)
            img_label = int(get_label(im_item))
            imgs.append((img_fn, img_label,))

        self.imgs = imgs
        self.filters = filters
        self.should_crop = should_crop and \
                           (not Conf.RunParams.DataLoader.
                            DISK_IMAGES_ALREADY_CROPPED)
        self.should_save_img_after_filters = should_save_img_after_filters

        # Calculate pre-filtered folder prefix
        self.filters_prefix = fil_to_str(self.filters)

        # Add field to track how many times filters are applied
        self.img_count_filters_applied = 0

    def __getitem__(self, index):
        """
        Loads an image and it's label
        - Applies the filters specified
        - Crops to only the face if enabled required
        :param index: Index of the image being requested
        :return: Image as a tensor and its label
        """
        im_path, im_label = self.imgs[index]
        org_fn = LocFolders.BASE_WORKING_DATASET_FOLDER + im_path
        fil_fn = self._get_precomputed_img_fn(im_path)
        if os.path.exists(fil_fn):
            # Load pre-filtered image (Already copped never crop)
            im_data = read_img_from_disk(fil_fn, False)
        else:
            im_data = read_img_from_disk(org_fn, self.should_crop)

            # Record that filters had to be applied
            self.img_count_filters_applied = self.img_count_filters_applied + 1

            # Apply filters
            for fil_class in self.filters:
                im_data = fil_class().execute(im_data)

            if self.should_save_img_after_filters:
                ensure_folder_created(fil_fn)
                transforms.ToPILImage()(im_data).save(fil_fn)
        return im_data, im_label

    def __len__(self):
        """
        Compute the number of images.
        :return: number of images
        """
        return len(self.imgs)

    def precompute(self):
        if len(self.filters) == 0:
            log("No filters so no need to precompute")
        elif self.is_precomputed():
            log("Already precomputed")
        elif not self.should_save_img_after_filters:
            # Request to precompute should not be called if saving is disabled
            log("Saving of precomputed images not enabled.", logging.ERROR)
        else:
            # Perform precompute

            sw = StopWatch(
                f'PreComputing Dataset for Filter(s): "{self.filters_prefix}"')
            # Load all images to ensure they are precomputed
            for i in range(len(self)):
                _ = self[i]
            sw.end()

    def is_precomputed(self):
        """
        Assumes that if one image is precomputed then all are so only test
        first image
        :return:
        """
        im_path, _ = self.imgs[0]
        return os.path.exists(self._get_precomputed_img_fn(im_path))

    def _get_precomputed_img_fn(self, im_path):
        return LocFolders.BASE_WORKING_DATASET_FOLDER + self.filters_prefix \
               + im_path
