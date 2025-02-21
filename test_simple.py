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
from data_create import data_create

def TIN_get_accuracy(args,kwargs):
    tin =TIN(args,**kwargs)
    tin.run()
    accuracy = tin.get_accuracy()
    return accuracy



if __name__ == '__main__':

    start_time = time.time()

    dir = './datasets/TIN'

    task_range = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
    worker_range = [50,100, 150, 200, 250, 300, 350, 400, 450, 500]
    domain_range = [5,6,7,8,9,10,11,12,13,14,15]
    #data_create(10000, 500, 10, dir)


    num_task = 10000
    num_worker = 100
    num_domain = 10
    max_iter = 10

    # test module
    task_range = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    worker_range = [5,10, 15, 20, 25, 30, 35, 40, 45, 50]
    domain_range = [5,6,7,8,9,10,11,12,13,14,15]
    num_worker = 100
    num_domain = 10
    max_iter = 10

    process = {}
    result = {}
    pool = Pool()
    for num_task in task_range:
        process[num_task] = {}
        process[num_task]['simple'] = {}
        process[num_task]['task_community'] = {}
        process[num_task]['worker_community'] = {}
        process[num_task]['full'] = {}

        result[num_task] = {}
        result[num_task]['simple'] = []
        result[num_task]['task_community'] = []
        result[num_task]['worker_community'] = []
        result[num_task]['full'] = []
        for iter in range(max_iter):
            dir_iter = dir + '/num_task' + str(num_task) + 'num_worker' + str(num_worker) + 'num_domain' + str(num_domain) + 'iter' + str(iter)
            datafile, truth_file, task_community_file, worker_community_file = data_create(num_task, num_worker,num_domain, dir_iter)

            process[num_task]['simple'][iter] = pool.apply_async(TIN_get_accuracy, args=(datafile, {'truth_file':truth_file, 'beta_1':1/3, 'beta_2':1/3, 'beta_3':1/3}))
            process[num_task]['task_community'][iter] = pool.apply_async(TIN_get_accuracy, args=(
            datafile, {'truth_file': truth_file, 'beta_1': 1 / 3, 'beta_2': 1 / 3, 'beta_3': 1 / 3, 'task_community_file':task_community_file}))
            process[num_task]['worker_community'][iter] = pool.apply_async(TIN_get_accuracy, args=(
                datafile, {'truth_file': truth_file, 'beta_1': 1 / 3, 'beta_2': 1 / 3, 'beta_3': 1 / 3,
                           'worker_community_file': worker_community_file}))
            process[num_task]['full'][iter] = pool.apply_async(TIN_get_accuracy, args=(
                datafile, {'truth_file': truth_file, 'beta_1': 1 / 3, 'beta_2': 1 / 3, 'beta_3': 1 / 3, 'task_community_file':task_community_file,
                           'worker_community_file': worker_community_file}))
            # tin = TIN(datafile, truth_file=truth_file, beta_1=1/3, beta_2=1/3, beta_3=1/3,task_community_file=task_community_file)
            # tin.run()
            # accuracy = tin.get_accuracy()
            # result[num_task]['task_community'].append(accuracy)
            # print('The accuracy of TIN_task_community is %f' % accuracy)
            #
            # tin = TIN(datafile, truth_file=truth_file, beta_1=1/3, beta_2=1/3, beta_3=1/3, worker_community_file=worker_community_file)
            # tin.run()
            # accuracy = tin.get_accuracy()
            # result[num_task]['worker_community'].append(accuracy)
            # print('The accuracy of TIN_worker_community is %f' % accuracy)
            #
            # tin = TIN(datafile, truth_file=truth_file, beta_1=1/3, beta_2=1/3, beta_3=1/3, task_community_file=task_community_file,worker_community_file=worker_community_file)
            # tin.run()
            # accuracy = tin.get_accuracy()
            # result[num_task]['full'].append(accuracy)
            # print('The accuracy of TIN_full is %f' % accuracy)
            # end_time = time.time()
            # print('time cost %f with num_task %d in iteration %d' % (end_time-start_time, num_task, iter))
    for num_task in task_range:
        for iter in range(max_iter):
            accuracy = process[num_task]['simple'][iter].get()
            result[num_task]['simple'].append(accuracy)
            print('The accuracy of TIN_simple is %f' % accuracy)

            accuracy = process[num_task]['task_community'][iter].get()
            result[num_task]['task_community'].append(accuracy)
            print('The accuracy of TIN_task is %f' % accuracy)

            accuracy = process[num_task]['worker_community'][iter].get()
            result[num_task]['worker_community'].append(accuracy)
            print('The accuracy of TIN_worker is %f' % accuracy)

            accuracy = process[num_task]['full'][iter].get()
            result[num_task]['full'].append(accuracy)
            print('The accuracy of TIN_full is %f' % accuracy)

    data_dir = dir + '/result_with_num_task.csv'
    with open(data_dir, 'w', newline='' , encoding='UTF-8') as f:
        writer = csv.writer(f)
        writer.writerow(['num_task', 'simple', 'simple_var', 'task_community','task_community_var', 'worker_community','worker_community_var', 'full', 'full_var'])
        for num_task in task_range:
            data = [num_task]
            data.append(np.mean(result[num_task]['simple']))
            data.append(np.var(result[num_task]['simple']))
            data.append(np.mean(result[num_task]['task_community']))
            data.append(np.var(result[num_task]['task_community']))
            data.append(np.mean(result[num_task]['worker_community']))
            data.append(np.var(result[num_task]['worker_community']))
            data.append(np.mean(result[num_task]['full']))
            data.append(np.var(result[num_task]['full']))
            writer.writerow(data)

    pool.close()
    pool.join()

    #TEST


