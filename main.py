from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.optimization import AdamW
from model_params import *
from dataset import ShakespearianDataset
from train import trainer

######### DATA FILE PATHS ##############
TEST_MODERN_PATH = "./data/test.modern.nltktok"
TEST_ORIGINAL_PATH = "./data/test.original.nltktok"

VAL_MODERN_PATH = "./data/valid.modern.nltktok"
VAL_ORIGINAL_PATH = "./data/valid.original.nltktok"

TRAIN_MODERN_PATH = "./data/train.modern.nltktok"
TRAIN_ORIGINAL_PATH = "./data/train.original.nltktok"

########### CHANGE BASED ON MODEL ############
params = bart_model_params

for k, v in params.items():
    print(k, ":\t", v)
######################################

if __name__ == "__main__":

    model = None
    tokenizer = None

    if "bart" in bart_model_params[MODEL]:
        model = BartForConditionalGeneration.from_pretrained(params[MODEL])
        tokenizer = BartTokenizer.from_pretrained(params[MODEL])

    else:
        raise Exception("Unknown model")

    original_vocab_length = len(tokenizer)
    train_set = ShakespearianDataset(source_file_path=TRAIN_MODERN_PATH,
                                     target_file_path=TRAIN_ORIGINAL_PATH,
                                     tokenizer=tokenizer,
                                     max_source_len=params[MAX_SOURCE_TEXT_LENGTH],
                                     max_target_len=params[MAX_TARGET_TEXT_LENGTH],
                                     add_tokens=True)

    # update the embedding layer in the model with new augmented vocab
    if len(tokenizer) > original_vocab_length:
        model.resize_token_embeddings(len(tokenizer))

    val_set = ShakespearianDataset(VAL_MODERN_PATH, VAL_ORIGINAL_PATH, tokenizer, params[MAX_SOURCE_TEXT_LENGTH],
                                   params[MAX_TARGET_TEXT_LENGTH])

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=params[TRAIN_BATCH_SIZE],
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=params[TRAIN_BATCH_SIZE],
        shuffle=True,
        num_workers=0
    )

    if "bart" in params[MODEL]:
        optimizer = AdamW(
            model.parameters(),
            lr=params[LEARNING_RATE],
        )
    else:
        raise Exception("code shouldn't have reached here!")


    # train the model
    trainer(model, tokenizer, optimizer, train_loader, val_loader, params)