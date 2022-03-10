MODEL = "MODEL"
TRAIN_BATCH_SIZE = "TRAIN_BATCH_SIZE"
VALID_BATCH_SIZE = "VALID_BATCH_SIZE"
TRAIN_EPOCHS = "TRAIN_EPOCHS"
VAL_EPOCHS = "VAL_EPOCHS"
LEARNING_RATE = "LEARNING_RATE"
MAX_SOURCE_TEXT_LENGTH = "MAX_SOURCE_TEXT_LENGTH"
MAX_TARGET_TEXT_LENGTH = "MAX_TARGET_TEXT_LENGTH"
SEED = "SEED"
OUTPUT_DIR = "OUTPUT_DIR"

bart_model_params = {
    MODEL: "facebook/bart-base",
    TRAIN_BATCH_SIZE: 8,
    VALID_BATCH_SIZE: 32,
    TRAIN_EPOCHS: 10,
    VAL_EPOCHS: 1,
    LEARNING_RATE: 3e-5,
    MAX_SOURCE_TEXT_LENGTH: 256,
    MAX_TARGET_TEXT_LENGTH: 256,
    SEED: 42,
    OUTPUT_DIR: "./output"
}