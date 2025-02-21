import csv
import xlwt
from operator import methodcaller
from Methods.BCC.method import BCC
from Methods.CATD.method import CATD
import Methods.CATD.read_distribution as cdis
from Methods.CBCC.method import CBCC
from Methods.DS.method import DS
from Methods.GLAD.method import GLAD
from Methods.MV.method import MV
from Methods.PM.method import PM
from Methods.TIN.method_linear import TIN
from Methods.ZC.method import ZC
from multiprocessing import Pool
import attr
import numpy as np
import random


def get_method_accuracy_in_data(func_name, datafile, truth_file, beta_1, beta_2, beta_3, label_relation_file):
    func = globals()[func_name]
    cls = func(datafile, truth_file=truth_file, beta_1=beta_1, beta_2=beta_2, beta_3=beta_3, label_relation_file=label_relation_file)
    cls.run()
    accuracy = cls.get_accuracy()
    print('The accuracy of %s on data %s  is %f' % (func, datafile, accuracy))

    return accuracy


# dataset_list = ['CF', 'CF_amt', 'd_Duck_Identification', 'd_jn_product', 'd_sentiment', 'MS', 's4_Dog_data',
#                 's4_Face_Sentiment_Identification', 's4_Relevance', 's5_AdultContent', 'SP', 'SP_amt', 'ZenCrowd_all',
#                 'ZenCrowd_in', 'ZenCrowd_us', 'MUSIC']
dataset_list = ['CF']
method_name = 'TIN'

f = xlwt.Workbook(encoding='ascii')
fs = f.add_sheet('accuracy_TIN', cell_overwrite_ok=True)
fs.write(0, 0, label='data')
fs.write(0, 1, label='beta1')
fs.write(0, 2, label='beta2')
fs.write(0, 3, label='beta3')
fs.write(0, 4, label='accuracy')

for i in range(len(dataset_list)):
    fs.write(i + 1, 0, label=dataset_list[i])

results = {}
pool = Pool()
for i in range(len(dataset_list)):
    dataset = dataset_list[i]
    datafile = './datasets/' + dataset + '/answer.csv'
    truth_file = './datasets/' + dataset + '/truth.csv'
    label_relation_file = './datasets/' + dataset + '/label_relation.csv'
    datafile_test = './datasets/' + dataset + '/answer_test.csv'
    datafile_validation = './datasets/' + dataset + '/answer_validation.csv'
    results[dataset] = {}
    # with open(datafile, 'r') as file:
    #     reader = csv.reader(file)
    #     head = next(reader)
    #     file_test = open(datafile_test, 'w', newline='')
    #     writer_test = csv.writer(file_test)
    #     writer_test.writerow(head)
    #     file_validation = open(datafile_validation, 'w', newline='')
    #     writer_validation = csv.writer(file_validation)
    #     writer_validation.writerow(head)
    #     for line in reader:
    #         if random.random() < 0.5:
    #             writer_validation.writerow(line)
    #         else:
    #             writer_test.writerow(line)
    #     file_test.close()
    #     file_validation.close()

    for beta_1 in np.linspace(0, 1, 21):
        results[dataset][beta_1] = {}
        for beta_2 in np.linspace(0, 1 - beta_1, int(round((1.05 - beta_1) / 0.05))):
            beta_3 = 1 - beta_1 - beta_2
            results[dataset][beta_1][beta_2] = pool.apply_async(get_method_accuracy_in_data,
                                                                args=(method_name, datafile_validation, truth_file, beta_1, beta_2, beta_3, label_relation_file))
results_test = {}
for i in range(len(dataset_list)):
    dataset = dataset_list[i]
    datafile = './datasets/' + dataset + '/answer.csv'
    truth_file = './datasets/' + dataset + '/truth.csv'
    datafile_test = './datasets/' + dataset + '/answer_test.csv'
    datafile_validation = './datasets/' + dataset + '/answer_validation.csv'
    label_relation_file = './datasets/' + dataset + '/label_relation.csv'
    opt_accuracy = 0
    opt_beta_1 = 0.2
    opt_beta_2 = 0.1
    opt_beta_3 = 0.7
    for beta_1 in np.linspace(0, 1, 21):
        results[beta_1] = {}
        for beta_2 in np.linspace(0, 1 - beta_1, int(round((1.05 - beta_1) / 0.05))):
            beta_3 = 1 - beta_1 - beta_2
            local_accuracy = results[dataset][beta_1][beta_2].get()
            if local_accuracy > opt_accuracy:
                opt_beta_1 = beta_1
                opt_beta_2 = beta_2
                opt_beta_3 = beta_3
                opt_accuracy = local_accuracy
    fs.write(i + 1, 1, label=opt_beta_1)
    fs.write(i + 1, 2, label=opt_beta_2)
    fs.write(i + 1, 3, label=opt_beta_3)
    results_test[dataset] = pool.apply_async(get_method_accuracy_in_data, args=(method_name, datafile_test, truth_file, opt_beta_1, opt_beta_2, opt_beta_3,label_relation_file))
for i in range(len(dataset_list)):
    dataset = dataset_list[i]
    accuracy = results_test[dataset].get()
    fs.write(i+1, 4, label=accuracy)

f.save('./accuracy_TIN.xls')
pool.close()
pool.join()
