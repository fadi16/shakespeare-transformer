import torch
import os.path
import pandas as pd
from rich import box
from rich.table import Table, Column

from generate import generate
from model_params import *
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import rich
from eval_metrics import *


def trainer(model, tokenizer, optimizer, training_loader, validation_loader, params):
    # for reproducibility
    set_seed(params)

    tb = SummaryWriter()

    # send to GPU/TPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("USING DEVICE " + device)
    model = model.to(device)

    training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII,
    )
    best_val_score = -1

    validate_step(-1, tokenizer, model, device, validation_loader, params, tb)

    for training_epoch in range(params[TRAIN_EPOCHS]):
        print("STARTING TRAINING EPOCH: " + str(training_epoch) + "\n")

        loss = train_step(epoch=training_epoch,
                          tokenizer=tokenizer,
                          model=model,
                          device=device,
                          loader=training_loader,
                          optimizer=optimizer,
                          logger=training_logger)
        tb.add_scalar("Loss", loss, training_epoch)

        # evaluate at the end of each epoch
        print("Validating after training epoch #{0}\n".format(str(training_epoch)))
        for validation_epoch in range(params[VAL_EPOCHS]):

            eval_score = validate_step(training_epoch, tokenizer, model, device, validation_loader, params, tb)
            print("overall bleurt score = ", eval_score)
            tb.add_scalar("overall_bleurt_score", eval_score, training_epoch)

            if eval_score > best_val_score:
                best_val_score = eval_score

            # save model and tokenizer
            model_checkpoint_path = os.path.join(params[OUTPUT_DIR], "checkpoint{0}".format(training_epoch))
            model.save_pretrained(model_checkpoint_path)
            tokenizer.save_pretrained(model_checkpoint_path)
            print("SAVED MODEL AT " + model_checkpoint_path + "\n")

            print("BEST BLEU SCORE = {0}, CURRENT BLEU SCORE = {1}\n".format(best_val_score, eval_score))


def train_step(epoch, tokenizer, model, device, loader, optimizer, logger):
    model.train()

    final_loss = None
    for _, data in enumerate(loader, start=0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        # In addition, we must make sure that padding token id’s of the labels are not taken into account by the loss function.
        # In PyTorch and Tensorflow, this can be done by replacing them with -100, which is the ignore_index of the CrossEntropyLoss
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels
        )

        # FA: this is cross entropy loss between predicted and golden output
        loss = outputs[0]
        final_loss = loss.item()

        if _ % 100 == 0:
            logger.add_row(str(epoch), str(_), str(loss))
            rich.print(logger)

        # clears old gradients from last step - so that they do not accumulate everytime you do loss.backwards
        optimizer.zero_grad()
        # back propagations
        loss.backward()
        # gradient decent
        optimizer.step()
    return final_loss


def validate_step(epoch, tokenizer, model, device, val_loader, params, tb: SummaryWriter):
    inputs, preds, refs = generate(tokenizer, model, device, val_loader, params)

    preds_vs_refs_df = pd.DataFrame({
        "input": inputs,
        "actual": refs,
        "generated": preds
    })

    preds_vs_refs_df.to_csv(os.path.join(params[OUTPUT_DIR], "generated_{0}.csv".format(epoch)))

    bleu2, bleu3, bleu4 = tuple(evaluate_bleu_micro_average(refs, preds))

    tb.add_scalar("bleu2", bleu2, epoch)
    tb.add_scalar("bleu3", bleu3, epoch)
    tb.add_scalar("bleu4", bleu4, epoch)

    print("**" * 5, " Validation Done ", "**" * 5)
    print("bleu2:\t", )
    print("**" * 15)

    return bleu4


def set_seed(model_params):
    torch.manual_seed(model_params[SEED])
    np.random.seed(model_params[SEED])
    torch.backends.cudnn.deterministic = True
