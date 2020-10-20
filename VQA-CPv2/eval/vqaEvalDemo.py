# coding: utf-8
import argparse
import sys

from vqaEval import VQAEval
import json
import random
import os

parser = argparse.ArgumentParser("Save a model's predictions for the VQA-CP test set")
parser.add_argument('--cp', action="store_true")
parser.add_argument('--resfile', type=str)
args = parser.parse_args()

if args.cp:
    from vqa import VQA
else:
    from vqa_orig import VQA

# set up file names and paths
versionType = 'v2_'  # this should be '' when using VQA v2.0 dataset
taskType = 'OpenEnded'  # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType = 'mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
dataSubType = 'train2014'
dataDir = '../data'
if args.cp:
    annFile = '../data/vqacp_v2_test_annotations.json'
    quesFile = '../data/vqacp_v2_test_questions.json'
else:
    annFile = '../data/v2_mscoco_val2014_annotations.json'
    quesFile = '../data/v2_OpenEnded_mscoco_val2014_questions.json'
resultType = 'fake'
fileTypes = ['results', 'accuracy', 'evalQA', 'evalQuesType', 'evalAnsType']

# An example result json file has been provided in './Results' folder.

[resFile, accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = [
    '%s%s_%s_%s_%s_%s.json' % (versionType, taskType, dataType, dataSubType, \
                                          resultType, fileType) for fileType in fileTypes]

resFile = args.resfile
# create vqa object and vqaRes object
vqa = VQA(annFile, quesFile)
vqaRes = vqa.loadRes(resFile, quesFile)

# create vqaEval object by taking vqa and vqaRes
vqaEval = VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2

# evaluate results
"""
If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
By default it uses all the question ids in annotation file
"""
vqaEval.evaluate()

# print accuracies
print "\n"
print "Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall'])
print "Per Question Type Accuracy is the following:"
for quesType in vqaEval.accuracy['perQuestionType']:
    print "%s : %.02f" % (quesType, vqaEval.accuracy['perQuestionType'][quesType])
print "\n"
print "Per Answer Type Accuracy is the following:"
for ansType in vqaEval.accuracy['perAnswerType']:
    print "%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType])
print "\n"
# demo how to use evalQA to retrieve low score result
evals = [quesId for quesId in vqaEval.evalQA if vqaEval.evalQA[quesId] < 35]  # 35 is per question percentage accuracy
if len(evals) > 0:
    print 'ground truth answers'
    randomEval = random.choice(evals)
    randomAnn = vqa.loadQA(randomEval)
    vqa.showQA(randomAnn)

    print '\n'
    print 'generated answer (accuracy %.02f)' % (vqaEval.evalQA[randomEval])
    ann = vqaRes.loadQA(randomEval)[0]
    print "Answer:   %s\n" % (ann['answer'])

# save evaluation results to ./Results folder
json.dump(vqaEval.accuracy, open(accuracyFile, 'w'))
json.dump(vqaEval.evalQA, open(evalQAFile, 'w'))
json.dump(vqaEval.evalQuesType, open(evalQuesTypeFile, 'w'))
json.dump(vqaEval.evalAnsType, open(evalAnsTypeFile, 'w'))
