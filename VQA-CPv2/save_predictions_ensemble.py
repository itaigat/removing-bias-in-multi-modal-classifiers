import argparse
import json
import pickle
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
    parser.add_argument('--cp', action="store_true")
    args = parser.parse_args()

    path = args.model

    print("Loading data...")
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    train_dset = VQAFeatureDataset('train', dictionary, cp=True, cache_image_features=True)
    eval_dset = VQAFeatureDataset('val', dictionary, cp=True, cache_image_features=True)

    eval_loader = DataLoader(eval_dset, 256, shuffle=False, num_workers=0)

    results = {}
    # models = ['model_0.5400412678718567.pth', 'model_0.5402570366859436.pth', 'model_0.5406695604324341.pth', 'model_0.5407657623291016.pth',
    #           'model_0.5409111976623535.pth', 'model_0.5413442254066467.pth', 'model_0.5418999791145325.pth', 'model_0.5422884821891785.pth',
    #           'model_0.5426487326622009.pth', 'model_0.543043315410614.pth', 'model_0.543769359588623.pth', 'model_0.5454666614532471.pth']

    models = ['model_0.5402570366859436.pth']

    for model_path in models:
        constructor = 'build_%s' % 'baseline0_newatt'
        model = getattr(base_model, constructor)(train_dset, 1024).cuda()

        state_dict = torch.load('saved_models/' + model_path)
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
            soft_pred = torch.nn.functional.softmax(factor, dim=1)
            if idx not in results:
                results[idx] = soft_pred.data.cpu()
            else:
                results[idx] += soft_pred.data.cpu()

            # prediction = torch.max(factor, 1)[1].data.cpu().numpy()
            # for p in prediction:
            #     predictions.append(train_dset.label2ans[p])

    max_results = []

    for key, item in results.items():
        prediction = torch.max(item, 1)[1].data.cpu().numpy()
        for p in prediction:
            max_results.append({'answer': train_dset.label2ans[p], 'question_id': key})

    pickle.dump(max_results, open(args.output_file + 'pkl', 'wb'))
    # out = []
    # for p, e in zip(max_results, eval_dset.entries):
    #     out.append(dict(answer=p, question_id=e["question_id"]))
    with open(args.output_file, "w") as f:
        json.dump(max_results, f)


if __name__ == '__main__':
    main()
