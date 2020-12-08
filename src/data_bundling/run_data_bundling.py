# General comments for file indicated expectations
"""
FORMAT_FILE = 'Data/train/5817/live/000012.jpg 0'
"""
import logging
import ntpath
import os
from shutil import copyfile
from typing import List
from zipfile import ZipFile

from torchvision.transforms import transforms

from src.dataset.text_file_rec_conversion import (get_bb_fn, get_filename,
                                                  get_label)
from src.dataset.the_dataset import read_img_from_disk
from src.main_ import finalize, initialize
from src.utils.config import Conf
from src.utils.log import log
from src.utils.misc_func import (change_filename_ext, mkdir_w_par,
                                 save_list_to_file)

conf = Conf.CreateBundle


def get_file_list(filename: str, req_count: int) -> List[str]:
    """
    Expected format of each line is as per FORMAT_LINE at top of file
    where the first part is the path to the file and the second is the label
    :param filename: text file with lines as per format above
    :param req_count: number of samples required from the file
    :return: a list of strings as per FORMAT_LINE at top of file
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    lines_count = len(lines)
    if req_count > lines_count:
        log(
            f'Input file has less lines than requested. Requested: '
            f'{req_count} Actual: {lines_count}',
            logging.ERROR)
        return lines
    if not Conf.CreateBundle.SHOULD_USE_REJECTION_SAMPLING:
        result = lines[0:req_count]
    else:
        result = []
        req_count_0 = req_count // 2
        req_count_1 = req_count - req_count_0
        count_0 = 0
        count_1 = 0
        count_total = 0  # Used to avoid need to add to find total
        is_req_count_met = False
        for line in lines:
            label = line[-2]
            if label == '0':
                if count_0 < req_count_0:
                    result.append(line)
                    count_0 += 1
                    count_total += 1
            elif label == '1':
                if count_1 < req_count_1:
                    result.append(line)
                    count_1 += 1
                    count_total += 1
            else:
                log(f'Unknown Label ({label}) found in {line}', logging.ERROR)
            if count_total >= req_count:
                is_req_count_met = True
                break
        if not is_req_count_met:
            log(
                f'Required count ({req_count}) not met. Exhausted File: '
                f'"{filename}" but only got 0\'s: {count_0}/{req_count_0} and '
                f'1\'s:  {count_1}/{req_count_1}',
                logging.ERROR)

    return result


def copy_img(old_fn: str, new_fn: str) -> str:
    """
    Copies the image from old_fn to new_fn. If configured it loads the image
    and crops it then saves the cropped version. To preserve quality the
    format is changed to png if this occurs, but the bounding box is not
    copied as it is no longer needed. Otherwise if it not loaded the file is
    simply copied along with the bounding box information
    :param old_fn: Filename to copy from
    :param new_fn: Filename to copy to
    :return: The new filename actually used (may change extension to
    preserve quality)
    """
    old_dir = conf.FileLocations.BASE_DATASET_FOLDER
    new_dir = conf.FileLocations.OUT_BUNDLE_FOLDER
    if Conf.CreateBundle.SHOULD_STORE_CROPPED:
        # Only store cropped version of image
        img = read_img_from_disk(old_dir + old_fn, True)
        img = transforms.ToPILImage()(img)
        new_fn = change_filename_ext(new_fn,
                                     Conf.CreateBundle.IMG_EXT_IF_SAVING)
        img.save(new_dir + new_fn)
    else:
        # Copy image file and bounding box file
        copyfile(old_dir + old_fn, new_dir + new_fn)
        copyfile(old_dir + get_bb_fn(old_fn), new_dir + get_bb_fn(new_fn))

    return new_fn  # Returned in the event that it needs to be changed


def copy_files(file_list: List[str], base_folder: str) -> List[str]:
    """
    Expects each line of file_list to be as per FORMAT_LINE at top of file
    :param file_list: a list with the files to copy
    :param base_folder: The folder to copy the files to
    :return: The list of files copied as per FORMAT_LINE at top of file
    """
    count_writen_files = 0
    count_folders = 0
    curr_folder = ''  # Initial value to be assigned by updater
    result = []

    def update_curr_folder():
        """
        Assumes that initially count_writen_files is 0 so that curr_folder
        gets set
        """
        nonlocal count_writen_files
        if count_writen_files % conf.Sizes.MAX_FILES_PER_FOLDER == 0:
            nonlocal count_folders, curr_folder
            count_folders += 1
            curr_folder = base_folder + '{:04d}'.format(count_folders) + '/'

    for rec in file_list:
        update_curr_folder()
        filename = get_filename(rec)
        label = get_label(rec)

        # Construct new filename
        remainder = filename
        remainder, new_filename = ntpath.split(remainder)  # Get filename
        remainder, sub_folder = ntpath.split(remainder)  # Get label folder
        new_folder = curr_folder + sub_folder + '/'
        remainder, temp = ntpath.split(remainder)  # Get id
        new_filename = temp + conf.FileLocations.OUT_ID_SEPARATOR + \
                       new_filename

        mkdir_w_par(conf.FileLocations.OUT_BUNDLE_FOLDER + new_folder)

        # Copy image and bounding box
        new_filename = new_folder + new_filename  # join folder with filename
        new_filename = copy_img(filename, new_filename)

        # Add to output list
        result.append(new_filename + ' ' + label)

        count_writen_files += 1
        if count_writen_files % conf.Sizes.NOTIFY_INTERVAL == 0:
            log(f'{count_writen_files} / {len(file_list)}')
    return result


def zip_output():
    # trim off trialing delimiter and add extension
    zip_filename, _ = ntpath.split(conf.FileLocations.OUT_BUNDLE_FOLDER[:-1])
    zip_filename += '/' + conf.FileLocations.OUT_ZIP_FILENAME_BASE + '.zip'
    base_dir = conf.FileLocations.OUT_BUNDLE_FOLDER
    root_len = len(base_dir)
    with ZipFile(zip_filename, 'w') as zipObj:
        # Iterate over all the files in directory
        for folderName, sub_folders, filenames in os.walk(base_dir):
            for filename in filenames:
                # create complete filepath of file in directory
                file_path = os.path.join(folderName, filename)
                # Add file to zip
                zipObj.write(file_path, file_path[root_len:])
    print(f'Zip file created at "{zip_filename}"')


def main():
    initialize()
    log("Main - Performing data bundling")
    assert conf.Sizes.TRAIN_COUNT > 0
    assert conf.Sizes.TEST_COUNT > 0

    # Confirm Empty output folder
    if os.path.exists(conf.FileLocations.OUT_BUNDLE_FOLDER):
        # Folder exists still ok if folder is empty
        if len(os.listdir(conf.FileLocations.OUT_BUNDLE_FOLDER)) > 0:
            raise Exception('Output folder is not Empty, Aborted!!!')

    # Get list of files to copy
    train_files = get_file_list(conf.FileLocations.TRAIN_DATA_INFO,
                                conf.Sizes.TRAIN_COUNT)
    test_files = get_file_list(conf.FileLocations.TEST_DATA_INFO,
                               conf.Sizes.TEST_COUNT)
    log(f'Train files to copy: {len(train_files)}')
    log(f'Test files to copy: {len(test_files)}')
    log('')  # To create a space between next output

    # Copy selected files
    log(f'Starting Training Set of {len(train_files)}')
    train_files = copy_files(train_files, conf.FileLocations.OUT_TRAIN_FOLDER)
    log(f'Starting Testing Set of {len(test_files)}')
    test_files = copy_files(test_files, conf.FileLocations.OUT_TEST_FOLDER)

    # Save file lists there
    save_list_to_file(train_files,
                      conf.FileLocations.OUT_BUNDLE_FOLDER +
                      conf.FileLocations.OUT_TRAIN_LIST_FILENAME)
    save_list_to_file(test_files,
                      conf.FileLocations.OUT_BUNDLE_FOLDER +
                      conf.FileLocations.OUT_TEST_LIST_FILENAME)

    zip_output()
    finalize()


if __name__ == '__main__':
    main()
