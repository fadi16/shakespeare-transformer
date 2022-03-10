import numpy as np
from nltk import RegexpTokenizer
from nltk.translate import bleu_score

BLEU2_INDEX = 1
BLEU3_INDEX = 1
BLEU4_INDEX = 1


def evaluate_bleu_micro_average(references, generated):
    regexp_tokenizer = RegexpTokenizer(r'\w+')

    references = [[regexp_tokenizer.tokenize(reference)] for reference in references]
    generated = [regexp_tokenizer.tokenize(generated_sample) for generated_sample in generated]
    bleu_scores = []
    weights = [(0.5, 0.5), (0.333, 0.333, 0.334), (0.25, 0.25, 0.25, 0.25)]
    for weight in weights:
        score = bleu_score.corpus_bleu(list_of_references=references, hypotheses=generated,
                                       smoothing_function=bleu_score.SmoothingFunction().method7,
                                       weights=weight)
        bleu_scores.append(score)

    return bleu_scores


def evaluate_bleu_macro_average(references, generated):
    regexp_tokenizer = RegexpTokenizer(r'\w+')

    references = [[regexp_tokenizer.tokenize(reference)] for reference in references]
    generated = [regexp_tokenizer.tokenize(generated_sample) for generated_sample in generated]
    bleu_scores = []
    weights = [(0.5, 0.5), (0.333, 0.333, 0.334), (0.25, 0.25, 0.25, 0.25)]
    for weight in weights:
        score = np.mean(
            [bleu_score.sentence_bleu(ref, gen, weight, bleu_score.SmoothingFunction().method7) for ref, gen in
             zip(references, generated)])
        bleu_scores.append(score)
    return score
