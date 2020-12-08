"""Simple file to show images from DataSet"""
from torchvision import transforms

from src.main_ import finalize, get_dataset, get_output_folder, initialize
from src.utils.config import Conf
from src.utils.enums import TDataUse
from src.utils.log import log
from src.utils.misc_func import mkdir_w_par, strs_to_classes


def main():
    should_display_img = False
    should_save_img = True
    stop_after_desired = True
    desired_img_count = 200
    initialize()
    log("Main - Starting Test")
    log(f"Running in test mode. Going to show {desired_img_count} images")
    filters_as_str = None  # If left as non defaults will be used
    if filters_as_str is None:
        filters_as_str = Conf.RunParams.MODEL_TRAIN_DEFAULTS['filters']
    filters = strs_to_classes(filters_as_str)
    log(f'Going to test: {filters_as_str}')
    dataset_train = get_dataset(TDataUse.TRAIN, filters)
    dataset_test = get_dataset(TDataUse.TEST, filters)
    log(
        f'There are {len(dataset_train)} training files and  '
        f'{len(dataset_test)} testing files')
    count_live = count_spoof = 0
    img_count = 0
    if should_save_img:
        # Make output folders
        live_save_folder = f'{get_output_folder()}live/'
        spoof_save_folder = f'{get_output_folder()}spoof/'
        mkdir_w_par(live_save_folder)
        mkdir_w_par(spoof_save_folder)
    for img, label in dataset_train:
        if label == 1:
            count_spoof += 1
        else:
            count_live += 1
        if img_count < desired_img_count:
            img = transforms.ToPILImage()(img)
            if should_display_img:
                img.show()
            if should_save_img:
                img.save(
                    f'{live_save_folder if label == 0 else spoof_save_folder}'
                    f'{img_count}.png')
            img_count += 1
        elif stop_after_desired:
            break
    log(
        f'There are {count_live} live images for training and {count_spoof} '
        f'spoof images')

    finalize()


if __name__ == '__main__':
    main()
