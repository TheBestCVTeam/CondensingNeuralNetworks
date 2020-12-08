import torch

from src.local.loc_folders import LocFolders


# Convention being used throughout code is to shorten names by using conf to
# represent values from the class needed in that context
class Conf:
    class RunParams:
        # Default values for tests
        MODEL_TRAIN_DEFAULTS = {
            'cnn_count': 4,
            'should_pretrain': True,
            'epochs': 20,
            'filters': [
                # 'src.filter.histeq.Histeq',
                # 'src.filter.gaussian.Gaussian',
                # 'src.filter.diffgaussian.DiffGaussian',
                # 'src.filter.diffusion.Diffusion',
            ],
            'dl_bypass_input': False,
            'learn_rate': 1e-5,
            'max_epochs_before_progress': 2,
            'max_total_epochs_no_progress': 5,
        }

        # This is a list of dicts. Each dict stores name value pairs for
        # TrainParam constructor specifying how to build the model (Overrides
        # defaults above).
        # NB: ID's are added to each dict based on their position in the list
        # at runtime
        MODEL_TRAIN = [

            # Defaults
            {},  # Pretrained model from paper with new fc layer, no filters
            {'cnn_count': 3},  # Drop last 1 CNN layer
            {'cnn_count': 2},  # Drop last 2 CNN layers
            {'cnn_count': 1},  # Drop last 3 CNN layers

            # Learning rate exploration
            # {'cnn_count': 1, 'learn_rate': 1e-2, },
            # {'cnn_count': 1, 'learn_rate': 1e-2, },
            # {'cnn_count': 1, 'learn_rate': 1e-3, },  # Standard learning rate
            # {'cnn_count': 1, 'learn_rate': 1e-3, },  # Standard learning rate
            # {'cnn_count': 1, 'learn_rate': 1e-4, },
            # {'cnn_count': 1, 'learn_rate': 1e-4, },
            # {'cnn_count': 1, 'learn_rate': 1e-5, },
            # {'cnn_count': 1, 'learn_rate': 1e-5, },
            # {'cnn_count': 1, 'learn_rate': 1e-6, },
            # {'cnn_count': 1, 'learn_rate': 1e-6, },

            # {'cnn_count': 1, 'learn_rate': 1e-2, },
            #
            # {'cnn_count': 1, 'learn_rate': 1e-4, },
            #
            # {'cnn_count': 1, 'learn_rate': 1e-5, },
            #
            # {'cnn_count': 1, 'learn_rate': 1e-2, 'epochs': 20},

            # # # Learning rate exploration + Diffusion
            # {'cnn_count': 1, 'learn_rate': 1e-2, 'filters': [
            # 'src.filter.diffusion.Diffusion', ], },
            # {'cnn_count': 1, 'learn_rate': 1e-4, 'filters': [
            # 'src.filter.diffusion.Diffusion', ]},
            # {'cnn_count': 1, 'learn_rate': 1e-5, 'filters': [
            # 'src.filter.diffusion.Diffusion', ]},
            # #
            # # # Learning rate exploration + Diffusion + random weights
            # {'cnn_count': 1, 'learn_rate': 1e-2, 'filters': [
            # 'src.filter.diffusion.Diffusion', ], 'should_pretrain': False,},
            # {'cnn_count': 1, 'learn_rate': 1e-4, 'filters': [
            # 'src.filter.diffusion.Diffusion', ], 'should_pretrain': False,},
            # {'cnn_count': 1, 'learn_rate': 1e-5, 'filters': [
            # 'src.filter.diffusion.Diffusion', ], 'should_pretrain': False,},
            #

            # With random weights
            {'should_pretrain': False},
            {'cnn_count': 3, 'should_pretrain': False},
            {'cnn_count': 2, 'should_pretrain': False},
            {'cnn_count': 1, 'should_pretrain': False},

            # # single filter tests - 4 cnn layers
            {'filters': ['src.filter.diffusion.Diffusion'], 'cnn_count': 4},
            {'filters': ['src.filter.gaussian.Gaussian'], 'cnn_count': 4},
            {'filters': ['src.filter.diffgaussian.DiffGaussian'],
             'cnn_count': 4},
            {'filters': ['src.filter.histeq.Histeq'], 'cnn_count': 4},
            #
            # # single filter tests - 3 cnn layers
            {'filters': ['src.filter.diffusion.Diffusion'], 'cnn_count': 3, },
            {'filters': ['src.filter.gaussian.Gaussian'], 'cnn_count': 3, },
            {'filters': ['src.filter.diffgaussian.DiffGaussian'],
             'cnn_count': 3},
            {'filters': ['src.filter.histeq.Histeq'], 'cnn_count': 3},

            # # single filter tests - 2 cnn layers
            {'filters': ['src.filter.diffusion.Diffusion'], 'cnn_count': 2},
            {'filters': ['src.filter.gaussian.Gaussian'], 'cnn_count': 2},
            {'filters': ['src.filter.diffgaussian.DiffGaussian'],
             'cnn_count': 2},
            {'filters': ['src.filter.histeq.Histeq'], 'cnn_count': 2},

            # # single filter tests - 1 cnn layers
            {'filters': ['src.filter.diffusion.Diffusion'], 'cnn_count': 1},
            {'filters': ['src.filter.gaussian.Gaussian'], 'cnn_count': 1},
            {'filters': ['src.filter.diffgaussian.DiffGaussian'],
             'cnn_count': 1},
            {'filters': ['src.filter.histeq.Histeq'], 'cnn_count': 1},

            # Fourier Test

            # With random weights
            # {"dl_bypass_input": True},
            # {'cnn_count': 3,"dl_bypass_input": True},
            # {'cnn_count': 2,"dl_bypass_input": True},
            # {'cnn_count': 1,"dl_bypass_input": True},

            # # single filter tests - 4 cnn layers
            # {'filters': ['src.filter.diffusion.Diffusion'], 'cnn_count':
            # 4, "dl_bypass_input": True},
            # {'filters': ['src.filter.gaussian.Gaussian'], 'cnn_count': 4,
            # "dl_bypass_input": True},
            # {'filters': ['src.filter.diffgaussian.DiffGaussian'],
            #  'cnn_count': 4, "dl_bypass_input": True},
            # {'filters': ['src.filter.histeq.Histeq'], 'cnn_count': 4,
            # "dl_bypass_input": True},

            # # # single filter tests - 3 cnn layers
            # {'filters': ['src.filter.diffusion.Diffusion'], 'cnn_count':
            # 3, "dl_bypass_input": True},
            # {'filters': ['src.filter.gaussian.Gaussian'], 'cnn_count': 3,
            # "dl_bypass_input": True},
            # {'filters': ['src.filter.diffgaussian.DiffGaussian'],
            #  'cnn_count': 3, "dl_bypass_input": True},
            # {'filters': ['src.filter.histeq.Histeq'], 'cnn_count': 3,
            # "dl_bypass_input": True},
            #
            # # # single filter tests - 2 cnn layers
            # {'filters': ['src.filter.diffusion.Diffusion'], 'cnn_count':
            # 2, "dl_bypass_input": True},
            # {'filters': ['src.filter.gaussian.Gaussian'], 'cnn_count': 2,
            # "dl_bypass_input": True},
            # {'filters': ['src.filter.diffgaussian.DiffGaussian'],
            #  'cnn_count': 2, "dl_bypass_input": True},
            # {'filters': ['src.filter.histeq.Histeq'], 'cnn_count': 2,
            # "dl_bypass_input": True},
            #
            # # # single filter tests - 1 cnn layers
            # {'filters': ['src.filter.diffusion.Diffusion'], 'cnn_count':
            # 1, "dl_bypass_input": True},
            # {'filters': ['src.filter.gaussian.Gaussian'], 'cnn_count': 1,
            # "dl_bypass_input": True},
            # {'filters': ['src.filter.diffgaussian.DiffGaussian'],
            #  'cnn_count': 1, "dl_bypass_input": True},
            # {'filters': ['src.filter.histeq.Histeq'], 'cnn_count': 1,
            # "dl_bypass_input": True},
        ]

        class DataLoader:
            # If there are issues with the data loader set this to 1 and try
            # again
            NUM_WORKERS = 4

            class BatchSizes:
                TRAIN = 500
                VALIDATION = 300
                TEST = 500

            DISK_IMAGES_ALREADY_CROPPED = True

        class Filters:
            class Gaussian:
                FILTER_SIZE_ORIG = 5
                SIGMA = 1
                CHANNELS = 3
                DIM = 2

            class DiffGaussian:
                FILTER_SIZE_ORIG = 5
                SIGMA1 = 4
                SIGMA2 = 2
                CHANNELS = 3
                DIM = 2

        TRAIN_MIN_PROGRESS = 1e-3
        SAVE_EACH_FINAL_MODEL = True

    class Training:
        ALLOW_GPU = True
        DTYPE = torch.float32

    class CreateBundle:
        SHOULD_USE_REJECTION_SAMPLING = True
        SHOULD_STORE_CROPPED = True
        IMG_EXT_IF_SAVING = '.png'

        class Sizes:
            TOTAL_COUNT = 50000
            TRAIN_SPLIT = .7  # Training percentage of total data size
            TRAIN_COUNT = round(TOTAL_COUNT * TRAIN_SPLIT)
            TEST_COUNT = TOTAL_COUNT - TRAIN_COUNT
            MAX_FILES_PER_FOLDER = 200
            NOTIFY_INTERVAL = 1000

        class FileLocations:
            BASE_DATASET_FOLDER = LocFolders.BASE_FULL_DATASET_FOLDER
            TRAIN_DATA_INFO = BASE_DATASET_FOLDER + \
                              'metas/intra_test/train_label.txt'
            TEST_DATA_INFO = BASE_DATASET_FOLDER + \
                             'metas/intra_test/test_label.txt'

            # Expected that all output is under this folder so a zip file
            # can be
            # made in this folder (Assumption used when creating zip)
            # It is also expected that this folder doesn't exist yet or is
            # empty
            OUT_BASE_FOLDER = LocFolders.BASE_BUNDLE_OUTPUT_FOLDER

            OUT_ZIP_FILENAME_BASE = 'CelebA_Spoof'
            OUT_BUNDLE_FOLDER = OUT_BASE_FOLDER + OUT_ZIP_FILENAME_BASE + '/'
            OUT_DATA_FOLDER = 'Data/'
            OUT_TRAIN_FOLDER = OUT_DATA_FOLDER + 'train/'
            OUT_TEST_FOLDER = OUT_DATA_FOLDER + 'test/'
            OUT_METAS_FOLDER = 'metas/'
            OUT_TRAIN_LIST_FILENAME = OUT_METAS_FOLDER + 'train_label.txt'
            OUT_TEST_LIST_FILENAME = OUT_METAS_FOLDER + 'test_label.txt'
            OUT_ID_SEPARATOR = '_'

    class DatasetInfo:
        SHOULD_RUN_ON_FULL_DATASET = False

    class Eval:
        ROC_FILENAME = 'roc.png'
        TRAINING_INFO_FILENAME = 'training.png'
        MODEL_AS_STR_FILENAME = 'model.txt'
        MODEL_FILENAME = 'model.pt'

    class Misc:
        OUTPUT_FOLDER = "output/"
        EVALUATIONS_FILENAME = 'evaluations.yaml'
        SETTINGS_FILENAME = 'settings.yaml'
        EVALUATIONS_CSV_FILENAME = 'evaluations.csv'
