# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-02-16 09:53:19
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2017-12-19 15:23:12

# from operator import add
#
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

from copy import deepcopy
from collections import namedtuple

from pprint import pprint

import pandas as pd

import sys

import numpy as np


## input as sentence level labels
def get_ner_fmeasure(name, golden_lists, predict_lists, experiment_dir_name, epoch, label_type="BMES", save_confusion_matrix=False):
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0
    golden_list_all = []
    predict_list_all = []

    for idx in range(0,sent_num):
        # word_list = sentence_lists[idx]
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]

        for g in golden_list:
            golden_list_all.append(g)

        for p in predict_list:
            predict_list_all.append(p)

        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)
        if label_type == "BMES":
            gold_matrix = get_ner_BMES(golden_list)
            pred_matrix = get_ner_BMES(predict_list)
        else:
            gold_matrix = get_ner_BIO(golden_list)
            pred_matrix = get_ner_BIO(predict_list)

        # print "gold", gold_matrix
        # print "pred", pred_matrix
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner

    if save_confusion_matrix:
        print('\nToken-level evaluation of the best performing model, at epoch', epoch, '\n')

        # Collect full entities
        y_gold = collect_named_entities(golden_list_all)
        y_pred = collect_named_entities(predict_list_all)

        num_classes = len(set(golden_list_all))

        if num_classes == 15:
            encoded_classes = ['B-PER', 'I-PER',
                       'B-ORG', 'I-ORG',
                       'B-GPE', 'I-GPE',
                       'B-LOC', 'I-LOC',
                       'B-DRV', 'I-DRV',
                       'B-EVT', 'I-EVT',
                       'B-PROD', 'I-PROD']

            targets = ['PER', 'ORG', 'GPE', 'LOC', 'DRV', 'EVT', 'PROD']

        elif num_classes == 13:
            encoded_classes = ['B-PER', 'I-PER',
                       'B-ORG', 'I-ORG',
                       'B-LOC', 'I-LOC',
                       'B-DRV', 'I-DRV',
                       'B-EVT', 'I-EVT',
                       'B-PROD', 'I-PROD']

            targets = ['PER', 'ORG', 'LOC', 'DRV', 'EVT', 'PROD']

        else:
            encoded_classes = ['B-PER', 'I-PER',
                       'B-ORG', 'I-ORG',
                       'B-GPE_ORG', 'I-GPE_ORG',
                       'B-LOC', 'I-LOC',
                       'B-GPE_LOC', 'I-GPE_LOC',
                       'B-DRV', 'I-DRV',
                       'B-EVT', 'I-EVT',
                       'B-PROD', 'I-PROD']

            targets = ['PER', 'ORG', 'GPE_ORG', 'LOC', 'GPE_LOC', 'DRV', 'EVT', 'PROD']

        results = compute_metrics(y_gold, y_pred, targets)

        print('Overall measures')
        print(pd.DataFrame.from_dict(results[0]), end='\n\n')


        for key, value in results[1:][0].items():
            print(key)
            print(pd.DataFrame.from_dict(value), end='\n\n')

        print('Strict evaluation:')
        strict_precision = results[0]['strict']['precision']
        strict_recall = results[0]['strict']['recall']

        strict_f1 = 2 * ((strict_precision * strict_recall) / (strict_precision + strict_recall))

        print('Precision', strict_precision)
        print('Recall', strict_recall)
        print('F1', strict_f1)

        cm = confusion_matrix(golden_list_all, predict_list_all, labels=encoded_classes)

        print(cm)

        cm[0, 0] = cm[0, 0] / 100

        clf_report = classification_report(golden_list_all, predict_list_all, labels=encoded_classes)

        macro_scores = precision_recall_fscore_support(golden_list_all, predict_list_all, labels=encoded_classes, average='macro')

        print(clf_report)

        print('\nMacro scores')
        print(macro_scores)

        cm_path = experiment_dir_name + '/confusion_matrix_' + name

        with open(cm_path + '.tsv', 'w') as f1:
            for row in cm:
                for element in row:
                    f1.write(str(element) + '\t')
                f1.write('\n')

        clf_report_path = experiment_dir_name + '/clf_report' + name

        with open(clf_report_path + '.txt', 'w') as f2:
            for row in clf_report.split('\n'):
                f2.write(row + '\n')

            f2.write('Macro\n')
            f2.write('Precision\tRecall\tF1\tSupport\n')
            f2.write('\t'.join([str(value) for value in macro_scores]))

        plt.clf()
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)


        # We want to show all ticks...
        ax.set_xticks(np.arange(len(encoded_classes)))
        ax.set_yticks(np.arange(len(encoded_classes)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(encoded_classes)
        ax.set_yticklabels(encoded_classes)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        threshold = cm.max() / 2

        # Loop over data dimensions and create text annotations.
        for i in range(len(encoded_classes)):
            for j in range(len(encoded_classes)):
                if i == j:
                    text = ax.text(j, i, cm[i, j], ha="center", va="center",
                                   fontstyle='oblique',
                                   color="white" if cm[i, j] > threshold else "black", fontsize=8)

                else:
                    text = ax.text(j, i, cm[i, j], ha="center", va="center",
                                   color="white" if cm[i, j] > threshold else "black", fontsize=8)

        plt.savefig(cm_path + '.png')
        print('\nSaved confusion matrix', cm_path)

    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)
    if predict_num == 0:
        precision = -1
    else:
        precision =  (right_num+0.0)/predict_num
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num+0.0)/golden_num
    if (precision == -1) or (recall == -1) or (precision+recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2*precision*recall/(precision+recall)
    accuracy = (right_tag+0.0)/all_tag
    # print "Accuracy: ", right_tag,"/",all_tag,"=",accuracy
    print("gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num)
    return accuracy, precision, recall, f_measure


def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string


def get_ner_BMES(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
            index_tag = current_label.replace(begin_label,"",1)

        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(single_label,"",1) +'[' +str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag +',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    # print stand_matrix
    return stand_matrix


def get_ner_BIO(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
                index_tag = current_label.replace(begin_label,"",1)
            else:
                tag_list.append(whole_tag + ',' + str(i-1))
                whole_tag = current_label.replace(begin_label,"",1)  + '[' + str(i)
                index_tag = current_label.replace(begin_label,"",1)

        elif inside_label in current_label:
            if current_label.replace(inside_label,"",1) == index_tag:
                whole_tag = whole_tag
            else:
                if (whole_tag != '')&(index_tag != ''):
                    tag_list.append(whole_tag +',' + str(i-1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '')&(index_tag != ''):
                tag_list.append(whole_tag +',' + str(i-1))
            whole_tag = ''
            index_tag = ''

    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix



def readSentence(input_file):
    in_lines = open(input_file,'r').readlines()
    sentences = []
    labels = []
    sentence = []
    label = []
    for line in in_lines:
        if len(line) < 2:
            sentences.append(sentence)
            labels.append(label)
            sentence = []
            label = []
        else:
            pair = line.strip('\n').split(' ')
            sentence.append(pair[0])
            label.append(pair[-1])
    return sentences,labels


def readTwoLabelSentence(input_file, pred_col=-1):
    in_lines = open(input_file,'r').readlines()
    sentences = []
    predict_labels = []
    golden_labels = []
    sentence = []
    predict_label = []
    golden_label = []
    for line in in_lines:
        if "##score##" in line:
            continue
        if len(line) < 2:
            sentences.append(sentence)
            golden_labels.append(golden_label)
            predict_labels.append(predict_label)
            sentence = []
            golden_label = []
            predict_label = []
        else:
            pair = line.strip('\n').split(' ')
            sentence.append(pair[0])
            golden_label.append(pair[1])
            predict_label.append(pair[pred_col])

    return sentences,golden_labels,predict_labels


def fmeasure_from_file(golden_file, predict_file, label_type="BMES"):
    print("Get f measure from file:", golden_file, predict_file)
    print("Label format:",label_type)
    golden_sent,golden_labels = readSentence(golden_file)
    predict_sent,predict_labels = readSentence(predict_file)
    P,R,F = get_ner_fmeasure(golden_labels, predict_labels, label_type)
    print ("P:%sm R:%s, F:%s"%(P,R,F))



def fmeasure_from_singlefile(twolabel_file, label_type="BMES", pred_col=-1):
    sent,golden_labels,predict_labels = readTwoLabelSentence(twolabel_file, pred_col)
    P,R,F = get_ner_fmeasure(golden_labels, predict_labels, label_type)
    print ("P:%s, R:%s, F:%s"%(P,R,F))


Entity = namedtuple("Entity", "e_type start_offset end_offset")


def collect_named_entities(tokens):
    """
    Creates a list of Entity named-tuples, storing the entity type and the start and end
    offsets of the entity.

    :param tokens: a list of labels
    :return: a list of Entity named-tuples
    """

    named_entities = []
    start_offset = None
    end_offset = None
    ent_type = None

    for offset, token_tag in enumerate(tokens):

        if token_tag == 'O':
            if ent_type is not None and start_offset is not None:
                end_offset = offset - 1
                named_entities.append(Entity(ent_type, start_offset, end_offset))
                start_offset = None
                end_offset = None
                ent_type = None

        elif ent_type is None:
            ent_type = token_tag[2:]
            start_offset = offset

        elif ent_type != token_tag[2:] or (ent_type == token_tag[2:] and token_tag[:1] == 'B'):

            end_offset = offset - 1
            named_entities.append(Entity(ent_type, start_offset, end_offset))

            # start of a new entity
            ent_type = token_tag[2:]
            start_offset = offset
            end_offset = None

    # catches an entity that goes up until the last token
    if ent_type and start_offset and end_offset is None:
        named_entities.append(Entity(ent_type, start_offset, len(tokens)-1))

    return named_entities


def compute_metrics(true_named_entities, pred_named_entities, targets):
    eval_metrics = {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurius': 0}
    target_tags_no_schema = targets

    # overall results
    evaluation = {'strict': deepcopy(eval_metrics),
                  'ent_type': deepcopy(eval_metrics),
                  'partial': deepcopy(eval_metrics),
                  'exact': deepcopy(eval_metrics)}

    # results by entity type
    evaluation_agg_entities_type = {e: deepcopy(evaluation) for e in target_tags_no_schema}

    true_which_overlapped_with_pred = []  # keep track of entities that overlapped

    # go through each predicted named-entity
    for pred in pred_named_entities:
        found_overlap = False

        # check if there's an exact match, i.e.: boundary and entity type match
        if pred in true_named_entities:
            true_which_overlapped_with_pred.append(pred)
            evaluation['strict']['correct'] += 1
            evaluation['ent_type']['correct'] += 1
            evaluation['exact']['correct'] += 1
            evaluation['partial']['correct'] += 1

            # for the agg. by e_type results
            evaluation_agg_entities_type[pred.e_type]['strict']['correct'] += 1
            evaluation_agg_entities_type[pred.e_type]['ent_type']['correct'] += 1

        else:

            # check for overlaps with any of the true entities
            for true in true_named_entities:

                # 1. check for an exact match but with a different e_type
                if true.start_offset == pred.start_offset and pred.end_offset == true.end_offset \
                        and true.e_type != pred.e_type:

                    # overall results
                    evaluation['strict']['incorrect'] += 1
                    evaluation['ent_type']['incorrect'] += 1
                    evaluation['partial']['correct'] += 1
                    evaluation['exact']['correct'] += 1

                    # aggregated by entity type results
                    evaluation_agg_entities_type[pred.e_type]['strict']['incorrect'] += 1
                    evaluation_agg_entities_type[pred.e_type]['ent_type']['incorrect'] += 1

                    true_which_overlapped_with_pred.append(true)
                    found_overlap = True
                    break

                # 2. check for an overlap i.e. not exact boundary match, with true entities
                elif pred.start_offset <= true.end_offset and true.start_offset <= pred.end_offset:

                    true_which_overlapped_with_pred.append(true)

                    # 2.1 overlaps with the same entity type
                    if pred.e_type == true.e_type:

                        # overall results
                        evaluation['strict']['incorrect'] += 1
                        evaluation['ent_type']['correct'] += 1
                        evaluation['partial']['partial'] += 1
                        evaluation['exact']['incorrect'] += 1

                        # aggregated by entity type results
                        evaluation_agg_entities_type[pred.e_type]['strict']['incorrect'] += 1
                        evaluation_agg_entities_type[pred.e_type]['ent_type']['correct'] += 1

                        found_overlap = True
                        break

                    # 2.2 overlaps with a different entity type
                    else:
                        # overall results
                        evaluation['strict']['incorrect'] += 1
                        evaluation['ent_type']['incorrect'] += 1
                        evaluation['partial']['partial'] += 1
                        evaluation['exact']['incorrect'] += 1

                        # aggregated by entity type results
                        evaluation_agg_entities_type[true.e_type]['strict']['missed'] += 1
                        evaluation_agg_entities_type[pred.e_type]['strict']['spurius'] += 1

                        found_overlap = True
                        break

            # count spurius (i.e., over-generated) entities
            if not found_overlap:
                # overall results
                evaluation['strict']['spurius'] += 1
                evaluation['ent_type']['spurius'] += 1
                evaluation['partial']['spurius'] += 1
                evaluation['exact']['spurius'] += 1

                # aggregated by entity type results
                evaluation_agg_entities_type[pred.e_type]['strict']['spurius'] += 1
                evaluation_agg_entities_type[pred.e_type]['ent_type']['spurius'] += 1

    # count missed entities
    for true in true_named_entities:
        if true in true_which_overlapped_with_pred:
            continue
        else:
            # overall results
            evaluation['strict']['missed'] += 1
            evaluation['ent_type']['missed'] += 1
            evaluation['partial']['missed'] += 1
            evaluation['exact']['missed'] += 1

            # for the agg. by e_type
            evaluation_agg_entities_type[true.e_type]['strict']['missed'] += 1
            evaluation_agg_entities_type[true.e_type]['ent_type']['missed'] += 1

    # Compute 'possible', 'actual', according to SemEval-2013 Task 9.1
    for eval_type in ['strict', 'ent_type', 'partial', 'exact']:
        correct = evaluation[eval_type]['correct']
        incorrect = evaluation[eval_type]['incorrect']
        partial = evaluation[eval_type]['partial']
        missed = evaluation[eval_type]['missed']
        spurius = evaluation[eval_type]['spurius']

        # possible: nr. annotations in the gold-standard which contribute to the final score
        evaluation[eval_type]['possible'] = correct + incorrect + partial + missed

        # actual: number of annotations produced by the NER system
        evaluation[eval_type]['actual'] = correct + incorrect + partial + spurius

        actual = evaluation[eval_type]['actual']
        possible = evaluation[eval_type]['possible']

        if eval_type in ['partial', 'ent_type']:
            precision = (correct + 0.5 * partial) / actual if actual > 0 else 0
            recall = (correct + 0.5 * partial) / possible if possible > 0 else 0
        else:
            precision = correct / actual if actual > 0 else 0
            recall = correct / possible if possible > 0 else 0

        evaluation[eval_type]['precision'] = precision
        evaluation[eval_type]['recall'] = recall

    return evaluation, evaluation_agg_entities_type


if __name__ == '__main__':
    # print "sys:",len(sys.argv)
    if len(sys.argv) == 3:
        fmeasure_from_singlefile(sys.argv[1],"BMES",int(sys.argv[2]))
    else:
        fmeasure_from_singlefile(sys.argv[1],"BMES")

