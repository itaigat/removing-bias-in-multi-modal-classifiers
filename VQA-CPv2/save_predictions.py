import argparse
import json
from os import listdir
from os.path import join, exists, isdir
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import base_model
from dataset import VQAFeatureDataset, Dictionary


def main():
    parser = argparse.ArgumentParser("Save a model's predictions for the VQA-CP test set")
    parser.add_argument("model", help="Directory of the model")
    parser.add_argument("output_file", help="File to write json output to")
    args = parser.parse_args()

    path = args.model

    print("Loading data...")
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    train_dset = VQAFeatureDataset('train', dictionary, cp=True, cache_image_features=True)
    eval_dset = VQAFeatureDataset('val', dictionary, cp=True, cache_image_features=True)

    eval_loader = DataLoader(eval_dset, 256, shuffle=False, num_workers=6)

    constructor = 'build_%s' % 'baseline0_newatt'
    model = getattr(base_model, constructor)(train_dset, 1024).cuda()

    print("Loading state dict for %s..." % path)

    state_dict = torch.load(join(path))
    if all(k.startswith("module.") for k in state_dict):
        filtered = {}
        for k in state_dict:
            filtered[k[len("module."):]] = state_dict[k]
        state_dict = filtered

    for k in list(state_dict):
        if k.startswith("debias_loss_fn"):
            del state_dict[k]

    model.load_state_dict(state_dict)

    model.cuda()
    model.eval()
    print("Done")

    predictions = []
    for v, q, a, b, idx in tqdm(eval_loader, ncols=100, total=len(eval_loader), desc="eval"):
        v = Variable(v, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        factor = model(v, None, q, None, None, True)[0]
        prediction = torch.max(factor, 1)[1].data.cpu().numpy()
        for p in prediction:
            predictions.append(train_dset.label2ans[p])

    out = []
    for p, e in zip(predictions, eval_dset.entries):
        out.append(dict(answer=p, question_id=e["question_id"]))
    with open(join(args.output_file), "w") as f:
        json.dump(out, f)


if __name__ == '__main__':
    main()