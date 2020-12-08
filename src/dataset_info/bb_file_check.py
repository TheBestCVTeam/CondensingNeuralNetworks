import cv2

from src.dataset.text_file_rec_conversion import get_filename
from src.dataset.the_dataset import perform_crop
from src.main_ import finalize, initialize
from src.utils.config import Conf
from src.utils.log import ErrorsStats, log
from src.utils.misc_func import (get_run_timestamp, mkdir_w_par,
                                 save_list_to_file)


# noinspection PyUnusedLocal
def test_file(filename):
    try:
        # Load image
        img = cv2.imread(filename)
        err_count_before = ErrorsStats.COUNT
        perform_crop(img, filename)
        return err_count_before == ErrorsStats.COUNT
    except Exception as e:
        log(f'Error testing Bounding Box of "{filename}". Msg: "{e}"')
        return False


def clean_list(in_filename, out_filename_base, base_in_folder):
    # Read input file
    with open(in_filename, 'r') as f:
        in_lines = f.readlines()
    log(f'{len(in_lines)} lines read from "{in_filename}"')

    # Prepare lists to store results
    good = []
    bad = []

    # Walk through input and separate input
    for line in in_lines:
        filename = get_filename(line)
        if test_file(base_in_folder + filename):
            good.append(line)
        else:
            bad.append(line)

    # Save output files
    save_list_to_file(good, out_filename_base + 'good.txt')
    save_list_to_file(bad, out_filename_base + 'bad.txt')


def main():
    initialize()
    file_loc = Conf.CreateBundle.FileLocations
    if Conf.DatasetInfo.SHOULD_RUN_ON_FULL_DATASET:
        base_in_folder = file_loc.BASE_DATASET_FOLDER
        train_filename = file_loc.TRAIN_DATA_INFO
        test_filename = file_loc.TEST_DATA_INFO
    else:
        base_in_folder = file_loc.OUT_BUNDLE_FOLDER
        train_filename = file_loc.OUT_BUNDLE_FOLDER + \
                         file_loc.OUT_TRAIN_LIST_FILENAME
        test_filename = file_loc.OUT_BUNDLE_FOLDER + \
                        file_loc.OUT_TEST_LIST_FILENAME

    out_folder = Conf.Misc.OUTPUT_FOLDER + get_run_timestamp() + '/'
    mkdir_w_par(out_folder)

    # Execute on Training Data
    clean_list(train_filename, out_folder + 'train_', base_in_folder)

    # Execute on Testing Data
    clean_list(test_filename, out_folder + 'test_', base_in_folder)

    finalize()


if __name__ == '__main__':
    main()
