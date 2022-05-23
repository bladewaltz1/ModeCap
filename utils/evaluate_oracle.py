import argparse
import json

import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from utils.logger import setup_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate")
    parser.add_argument("--gt_caption", type=str)
    parser.add_argument("--pd_caption", type=str)
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()

    logger = setup_logger("evaluate", args.save_dir, 0)
    ptb_tokenizer = PTBTokenizer()

    scorers = [(Cider(), "C"), (Spice(), "S"),
               (Bleu(4), ["B1", "B2", "B3", "B4"]),
               (Meteor(), "M"), (Rouge(), "R")]

    logger.info(f"loading ground-truths from {args.gt_caption}")
    with open(args.gt_caption) as f:
        gt_captions = json.load(f)
    gt_captions = ptb_tokenizer.tokenize(gt_captions)

    # logger.info(f"loading predictions from {args.pd_caption}")
    with open(args.pd_caption) as f:
        pd_captions = json.load(f)
    pd_captions = ptb_tokenizer.tokenize(pd_captions)

    num_activated_modes = len(list(pd_captions.values())[0])
    pd_captions_all_modes = [{k : [v[i]] for k, v in pd_captions.items()} 
                              for i in range(num_activated_modes)]

    logger.info("Start evaluating")
    score_all_modes = {"C": [], "S": [], "M": [], "R": [],
                       "B1": [], "B2": [], "B3": [], "B4": []}
    for i, pd_captions in enumerate(pd_captions_all_modes):
        scores = {}
        for (scorer, method) in scorers:
            score, score_list = scorer.compute_score(gt_captions, pd_captions)
            if type(score) == list:
                for m, s in zip(method, score):
                    scores[m] = s
            else:
                scores[method] = score

            if method == "C":
                score_all_modes["C"].append(np.asarray(score_list))
            elif method == "S":
                tmp = [item["All"]["f"] for item in score_list]
                score_all_modes["S"].append(np.asarray(tmp))
            elif "B1" in method:
                score_all_modes["B1"].append(np.asarray(score_list[0]))
                score_all_modes["B2"].append(np.asarray(score_list[1]))
                score_all_modes["B3"].append(np.asarray(score_list[2]))
                score_all_modes["B4"].append(np.asarray(score_list[3]))
            elif method == "M":
                score_all_modes["M"].append(np.asarray(score_list))
            elif method == "R":
                score_all_modes["R"].append(np.asarray(score_list))

        logger.info(
            ' '.join([
                "C: {C:.4f}", "S: {S:.4f}",
                "M: {M:.4f}", "R: {R:.4f}",
                "B1: {B1:.4f}", "B2: {B2:.4f}",
                "B3: {B3:.4f}", "B4: {B4:.4f}"
            ]).format(
                C=scores['C'], S=scores['S'],
                M=scores['M'], R=scores['R'],
                B1=scores['B1'], B2=scores['B2'],
                B3=scores['B3'], B4=scores['B4']
            ))

    for k, v in score_all_modes.items():
        k_all_modes = np.stack(v, axis=1)
        logger.info("oracle {k}: {kall:.4f}".format(
            k=k, kall=k_all_modes.max(axis=1).mean()
        ))
