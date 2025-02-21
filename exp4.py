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
    print('start_time is %f' % start_time)

    dir = './datasets/TIN'

    task_range = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    worker_range = [50,100, 150, 200, 250, 300, 350, 400, 450, 500]
    domain_range = [5,6,7,8,9,10,11,12,13,14,15]
    redundancy_range = [5,6,7,8,9,10,11,12,13,14,15]
    #data_create(10000, 500, 10, dir)


    num_task = 1000
    num_worker = 100
    num_domain = 10
    num_redundancy = 5
    max_iter = 20


    process = {}
    result = {}
    pool = Pool(40)
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
            dir_iter = dir + '/task_range_' +'num_task' + str(num_task) + 'num_worker' + str(num_worker) + 'num_domain' + str(num_domain) + 'num_redundancy' + str(num_redundancy) + 'iter' + str(iter)
            datafile, truth_file, task_community_file, worker_community_file = data_create(num_task, num_worker,num_domain, dir_iter, num_redundancy)

            process[num_task]['simple'][iter] = pool.apply_async(TIN_get_accuracy, args=(datafile, {'truth_file':truth_file, 'beta_1':1/3, 'beta_2':1/3, 'beta_3':1/3}))
            process[num_task]['task_community'][iter] = pool.apply_async(TIN_get_accuracy, args=(
            datafile, {'truth_file': truth_file, 'beta_1': 1 / 3, 'beta_2': 1 / 3, 'beta_3': 1 / 3, 'task_community_file':task_community_file}))
            process[num_task]['worker_community'][iter] = pool.apply_async(TIN_get_accuracy, args=(
                datafile, {'truth_file': truth_file, 'beta_1': 1 / 3, 'beta_2': 1 / 3, 'beta_3': 1 / 3,
                           'worker_community_file': worker_community_file}))
            process[num_task]['full'][iter] = pool.apply_async(TIN_get_accuracy, args=(
                datafile, {'truth_file': truth_file, 'beta_1': 1 / 3, 'beta_2': 1 / 3, 'beta_3': 1 / 3, 'task_community_file':task_community_file,
                           'worker_community_file': worker_community_file}))
    for num_task in task_range:
        for iter in range(max_iter):
            accuracy = process[num_task]['simple'][iter].get()
            result[num_task]['simple'].append(accuracy)
            print('The accuracy of TIN_full is %f' % accuracy)

            accuracy = process[num_task]['task_community'][iter].get()
            result[num_task]['task_community'].append(accuracy)
            print('The accuracy of TIN_full is %f' % accuracy)

            accuracy = process[num_task]['worker_community'][iter].get()
            result[num_task]['worker_community'].append(accuracy)
            print('The accuracy of TIN_full is %f' % accuracy)

            accuracy = process[num_task]['full'][iter].get()
            result[num_task]['full'].append(accuracy)
            print('The accuracy of TIN_full is %f' % accuracy)
    data_dir = dir + '/result_with_task_range.csv'
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

    process = {}
    result = {}
    pool = Pool(40)
    for num_worker in worker_range:
        process[num_worker] = {}
        process[num_worker]['simple'] = {}
        process[num_worker]['task_community'] = {}
        process[num_worker]['worker_community'] = {}
        process[num_worker]['full'] = {}

        result[num_worker] = {}
        result[num_worker]['simple'] = []
        result[num_worker]['task_community'] = []
        result[num_worker]['worker_community'] = []
        result[num_worker]['full'] = []
        for iter in range(max_iter):
            dir_iter = dir + '/worker_range_' + 'num_task' + str(num_task) + 'num_worker' + str(
                num_worker) + 'num_domain' + str(num_domain) + 'num_redundancy' + str(num_redundancy) + 'iter' + str(iter)
            datafile, truth_file, task_community_file, worker_community_file = data_create(num_task, num_worker,
                                                                                           num_domain, dir_iter,
                                                                                           num_redundancy)

            process[num_worker]['simple'][iter] = pool.apply_async(TIN_get_accuracy, args=(
            datafile, {'truth_file': truth_file, 'beta_1': 1 / 3, 'beta_2': 1 / 3, 'beta_3': 1 / 3}))
            process[num_worker]['task_community'][iter] = pool.apply_async(TIN_get_accuracy, args=(
                datafile, {'truth_file': truth_file, 'beta_1': 1 / 3, 'beta_2': 1 / 3, 'beta_3': 1 / 3,
                           'task_community_file': task_community_file}))
            process[num_worker]['worker_community'][iter] = pool.apply_async(TIN_get_accuracy, args=(
                datafile, {'truth_file': truth_file, 'beta_1': 1 / 3, 'beta_2': 1 / 3, 'beta_3': 1 / 3,
                           'worker_community_file': worker_community_file}))
            process[num_worker]['full'][iter] = pool.apply_async(TIN_get_accuracy, args=(
                datafile, {'truth_file': truth_file, 'beta_1': 1 / 3, 'beta_2': 1 / 3, 'beta_3': 1 / 3,
                           'task_community_file': task_community_file,
                           'worker_community_file': worker_community_file}))
    for num_worker in worker_range:
        for iter in range(max_iter):
            accuracy = process[num_worker]['simple'][iter].get()
            result[num_worker]['simple'].append(accuracy)
            print('The accuracy of TIN_full is %f' % accuracy)

            accuracy = process[num_worker]['task_community'][iter].get()
            result[num_worker]['task_community'].append(accuracy)
            print('The accuracy of TIN_full is %f' % accuracy)

            accuracy = process[num_worker]['worker_community'][iter].get()
            result[num_worker]['worker_community'].append(accuracy)
            print('The accuracy of TIN_full is %f' % accuracy)

            accuracy = process[num_worker]['full'][iter].get()
            result[num_worker]['full'].append(accuracy)
            print('The accuracy of TIN_full is %f' % accuracy)
    data_dir = dir + '/result_with_worker_range.csv'
    with open(data_dir, 'w', newline='', encoding='UTF-8') as f:
        writer = csv.writer(f)
        writer.writerow(['num_worker', 'simple', 'simple_var', 'task_community', 'task_community_var', 'worker_community',
                         'worker_community_var', 'full', 'full_var'])
        for num_worker in worker_range:
            data = [num_worker]
            data.append(np.mean(result[num_worker]['simple']))
            data.append(np.var(result[num_worker]['simple']))
            data.append(np.mean(result[num_worker]['task_community']))
            data.append(np.var(result[num_worker]['task_community']))
            data.append(np.mean(result[num_worker]['worker_community']))
            data.append(np.var(result[num_worker]['worker_community']))
            data.append(np.mean(result[num_worker]['full']))
            data.append(np.var(result[num_worker]['full']))
            writer.writerow(data)
    pool.close()
    pool.join()

    process = {}
    result = {}
    pool = Pool(40)
    for num_domain in domain_range:
        process[num_domain] = {}
        process[num_domain]['simple'] = {}
        process[num_domain]['task_community'] = {}
        process[num_domain]['worker_community'] = {}
        process[num_domain]['full'] = {}

        result[num_domain] = {}
        result[num_domain]['simple'] = []
        result[num_domain]['task_community'] = []
        result[num_domain]['worker_community'] = []
        result[num_domain]['full'] = []
        for iter in range(max_iter):
            dir_iter = dir + '/domain_range_' + 'num_task' + str(num_task) + 'num_worker' + str(
                num_worker) + 'num_domain' + str(num_domain)+ 'num_redundancy' + str(num_redundancy) + 'iter' + str(iter)
            datafile, truth_file, task_community_file, worker_community_file = data_create(num_task, num_worker,
                                                                                           num_domain, dir_iter,
                                                                                           num_redundancy)

            process[num_domain]['simple'][iter] = pool.apply_async(TIN_get_accuracy, args=(
                datafile, {'truth_file': truth_file, 'beta_1': 1 / 3, 'beta_2': 1 / 3, 'beta_3': 1 / 3}))
            process[num_domain]['task_community'][iter] = pool.apply_async(TIN_get_accuracy, args=(
                datafile, {'truth_file': truth_file, 'beta_1': 1 / 3, 'beta_2': 1 / 3, 'beta_3': 1 / 3,
                           'task_community_file': task_community_file}))
            process[num_domain]['worker_community'][iter] = pool.apply_async(TIN_get_accuracy, args=(
                datafile, {'truth_file': truth_file, 'beta_1': 1 / 3, 'beta_2': 1 / 3, 'beta_3': 1 / 3,
                           'worker_community_file': worker_community_file}))
            process[num_domain]['full'][iter] = pool.apply_async(TIN_get_accuracy, args=(
                datafile, {'truth_file': truth_file, 'beta_1': 1 / 3, 'beta_2': 1 / 3, 'beta_3': 1 / 3,
                           'task_community_file': task_community_file,
                           'worker_community_file': worker_community_file}))
    for num_domain in domain_range:
        for iter in range(max_iter):
            accuracy = process[num_domain]['simple'][iter].get()
            result[num_domain]['simple'].append(accuracy)
            print('The accuracy of TIN_full is %f' % accuracy)

            accuracy = process[num_domain]['task_community'][iter].get()
            result[num_domain]['task_community'].append(accuracy)
            print('The accuracy of TIN_full is %f' % accuracy)

            accuracy = process[num_domain]['worker_community'][iter].get()
            result[num_domain]['worker_community'].append(accuracy)
            print('The accuracy of TIN_full is %f' % accuracy)

            accuracy = process[num_domain]['full'][iter].get()
            result[num_domain]['full'].append(accuracy)
            print('The accuracy of TIN_full is %f' % accuracy)
    data_dir = dir + '/result_with_domain_range.csv'
    with open(data_dir, 'w', newline='', encoding='UTF-8') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['num_domain', 'simple', 'simple_var', 'task_community', 'task_community_var', 'worker_community',
             'worker_community_var', 'full', 'full_var'])
        for num_domain in domain_range:
            data = [num_domain]
            data.append(np.mean(result[num_domain]['simple']))
            data.append(np.var(result[num_domain]['simple']))
            data.append(np.mean(result[num_domain]['task_community']))
            data.append(np.var(result[num_domain]['task_community']))
            data.append(np.mean(result[num_domain]['worker_community']))
            data.append(np.var(result[num_domain]['worker_community']))
            data.append(np.mean(result[num_domain]['full']))
            data.append(np.var(result[num_domain]['full']))
            writer.writerow(data)
    pool.close()
    pool.join()

    process = {}
    result = {}
    pool = Pool(40)
    for num_redundancy in redundancy_range:
        process[num_redundancy] = {}
        process[num_redundancy]['simple'] = {}
        process[num_redundancy]['task_community'] = {}
        process[num_redundancy]['worker_community'] = {}
        process[num_redundancy]['full'] = {}

        result[num_redundancy] = {}
        result[num_redundancy]['simple'] = []
        result[num_redundancy]['task_community'] = []
        result[num_redundancy]['worker_community'] = []
        result[num_redundancy]['full'] = []
        for iter in range(max_iter):
            dir_iter = dir + '/redundancy_range_' + 'num_task' + str(num_task) + 'num_worker' + str(
                num_worker) + 'num_domain' + str(num_domain)+ 'num_redundancy' + str(num_redundancy) + 'iter' + str(iter)
            datafile, truth_file, task_community_file, worker_community_file = data_create(num_task, num_worker,
                                                                                           num_domain, dir_iter,
                                                                                           num_redundancy)

            process[num_redundancy]['simple'][iter] = pool.apply_async(TIN_get_accuracy, args=(
                datafile, {'truth_file': truth_file, 'beta_1': 1 / 3, 'beta_2': 1 / 3, 'beta_3': 1 / 3}))
            process[num_redundancy]['task_community'][iter] = pool.apply_async(TIN_get_accuracy, args=(
                datafile, {'truth_file': truth_file, 'beta_1': 1 / 3, 'beta_2': 1 / 3, 'beta_3': 1 / 3,
                           'task_community_file': task_community_file}))
            process[num_redundancy]['worker_community'][iter] = pool.apply_async(TIN_get_accuracy, args=(
                datafile, {'truth_file': truth_file, 'beta_1': 1 / 3, 'beta_2': 1 / 3, 'beta_3': 1 / 3,
                           'worker_community_file': worker_community_file}))
            process[num_redundancy]['full'][iter] = pool.apply_async(TIN_get_accuracy, args=(
                datafile, {'truth_file': truth_file, 'beta_1': 1 / 3, 'beta_2': 1 / 3, 'beta_3': 1 / 3,
                           'task_community_file': task_community_file,
                           'worker_community_file': worker_community_file}))
    for num_redundancy in redundancy_range:
        for iter in range(max_iter):
            accuracy = process[num_redundancy]['simple'][iter].get()
            result[num_redundancy]['simple'].append(accuracy)
            print('The accuracy of TIN_full is %f' % accuracy)

            accuracy = process[num_redundancy]['task_community'][iter].get()
            result[num_redundancy]['task_community'].append(accuracy)
            print('The accuracy of TIN_full is %f' % accuracy)

            accuracy = process[num_redundancy]['worker_community'][iter].get()
            result[num_redundancy]['worker_community'].append(accuracy)
            print('The accuracy of TIN_full is %f' % accuracy)

            accuracy = process[num_redundancy]['full'][iter].get()
            result[num_redundancy]['full'].append(accuracy)
            print('The accuracy of TIN_full is %f' % accuracy)
    data_dir = dir + '/result_with_redundancy_range.csv'
    with open(data_dir, 'w', newline='', encoding='UTF-8') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['num_redundancy', 'simple', 'simple_var', 'task_community', 'task_community_var', 'worker_community',
             'worker_community_var', 'full', 'full_var'])
        for num_redundancy in redundancy_range:
            data = [num_redundancy]
            data.append(np.mean(result[num_redundancy]['simple']))
            data.append(np.var(result[num_redundancy]['simple']))
            data.append(np.mean(result[num_redundancy]['task_community']))
            data.append(np.var(result[num_redundancy]['task_community']))
            data.append(np.mean(result[num_redundancy]['worker_community']))
            data.append(np.var(result[num_redundancy]['worker_community']))
            data.append(np.mean(result[num_redundancy]['full']))
            data.append(np.var(result[num_redundancy]['full']))
            writer.writerow(data)
    pool.close()
    pool.join()

    end_time = time.time()
    print('end_time is %f' % end_time)
    print('time cost is %f' % (end_time-start_time))

    #TEST


