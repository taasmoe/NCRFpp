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

from sklearn.metrics import confusion_matrix, classification_report

from time import gmtime, strftime

import sys

import numpy as np


## input as sentence level labels
def get_ner_fmeasure(name, golden_lists, predict_lists, experiment_dir_name, label_type="BMES", save_confusion_matrix=False):
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
        classes = ['O',
                   'B-PER', 'I-PER',
                   'B-ORG', 'I-ORG',
                   'B-GPE_ORG', 'I-GPE_ORG',
                   'B-LOC', 'I-LOC',
                   'B-GPE_LOC', 'I-GPE_LOC',
                   'B-DRV', 'I-DRV',
                   'B-EVT', 'I-EVT',
                   'B-PROD', 'I-PROD']

        cm = confusion_matrix(golden_list_all, predict_list_all, labels=classes)

        clf_report = classification_report(golden_list_all, predict_list_all, labels=classes)

        print(clf_report)

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


        fig, ax = plt.subplots()
        im = ax.imshow(cm)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(classes)):
            for j in range(len(classes)):
                text = ax.text(j, i, cm[i, j], ha="center", va="center", color="w", fontsize=8)

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



if __name__ == '__main__':
    # print "sys:",len(sys.argv)
    if len(sys.argv) == 3:
        fmeasure_from_singlefile(sys.argv[1],"BMES",int(sys.argv[2]))
    else:
        fmeasure_from_singlefile(sys.argv[1],"BMES")

