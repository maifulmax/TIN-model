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
import time


def get_method_accuracy_in_data(func_name, datafile, truth_file, beta_1, beta_2, beta_3, **kwargs):
    func = globals()[func_name]
    cls = func(datafile, truth_file=truth_file, beta_1=beta_1, beta_2=beta_2, beta_3=beta_3, **kwargs)
    cls.run()
    accuracy = cls.get_accuracy()
    print('The accuracy of %s on data %s  is %f' % (func, datafile, accuracy))

    return accuracy


start_time = time.time()

method_name = 'TIN'


results = {}
pool = Pool()

f = open('opt_bata.csv','r')
reader = csv.reader(f)
next(reader)

for dataset, beta_1, beta_2, beta_3 in reader:
    dataset = str(dataset)
    beta_1 = float(beta_1)
    beta_2 = float(beta_2)
    beta_3 = float(beta_3)
    datafile = './datasets/' + dataset + '/answer.csv'
    truth_file = './datasets/' + dataset + '/truth.csv'
    datafile_test = './datasets/' + dataset + '/answer_test.csv'
    datafile_validation = './datasets/' + dataset + '/answer_validation.csv'
    task_community_file = './datasets/' + dataset + '/task_relation.csv'
    worker_community_file = './datasets/' + dataset + '/worker_relation.csv'
    label_relation_file = './datasets/' + dataset + '/label_relation.csv'

    results[dataset] = pool.apply_async(get_method_accuracy_in_data, args=(
        method_name, datafile_test, truth_file, beta_1, beta_2, beta_3),
                                             kwds={})

f.close()

with open('exp3_TIN_accuracy_base.csv','w', newline='',encoding='UTF-8') as f2:
    writer = csv.writer(f2)
    writer.writerow(['dataset','accuracy'])
    for dataset in results:
        accuracy = results[dataset].get()
        writer.writerow([dataset,accuracy])

pool.close()
pool.join()

end_time = time.time()
total_time = (end_time - start_time) / 3600
print('time cost: %f' % total_time)
