EMBEDDING_SPACE = 256
IMG_DIM = [224, 224]


MODEL_DIR_PATH = "models/VGG16/VGG16_weights_hardTriplets.pth"
DATA_DIR_PATH = "Data/head/"

HARD_TRIPLET_PATH ="hard_triplets"

TRIPLETLOSS_MARGIN = 2
SPLIT_RATIO = 0.8
LEARNING_RATE = 1e-03
EPOCHES = 10
BATCH_SIZE = 10
HARD_TRIPLETS = False
HARD_TRIPLET_RESTRICTION = 10
RELLATIVE_LOSS_LIMIT = .1


OPPERATING_MODES = ['Validate', 'Plot', 'Train'] #
#'Plot', 'Validate', 'Save', 'Train', 'Load'