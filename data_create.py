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
import os


def data_create(num_tasks, num_workers, num_domain, directory , num_redundancy=5, **kwargs):
    num_skills = 6

    # task parameters
    main_skill_weight = 5
    common_skill_weight = 1

    # worker parameters
    main_skill_ability = 0.8
    common_skill_ability = 0.4

    if not os.path.exists(directory):
        os.makedirs(directory)

    datafile = directory+ '/answer.csv'
    truth_file = directory + '/truth.csv'
    task_community_file = directory + '/task_relation.csv'
    worker_community_file = directory + '/worker_relation.csv'

    for kw in kwargs:
        setattr(self, kw, kwargs[kw])

    task_list = range(num_tasks)
    worker_list = range(num_workers)
    label_domain = range(num_domain)


    skill_dirichlet_parameters = np.ones([num_skills, num_skills]) * common_skill_weight + np.eye(num_skills)*(main_skill_weight - common_skill_weight)
    # add a task category with full skill requirements
    tem = np.ones([1, num_skills])*main_skill_weight
    skill_dirichlet_parameters = np.append(skill_dirichlet_parameters, tem, axis=0)

    worker_skill_parameters = np.ones([num_skills, num_skills]) * common_skill_ability + np.eye(num_skills) * (
                main_skill_ability - common_skill_ability)
    # add a worker community with full skilled workers.
    tem = np.ones([1, num_skills]) * main_skill_ability
    worker_skill_parameters = np.append(worker_skill_parameters, tem, axis=0)
    # add a worker community with poor skilled workers.
    tem = np.ones([1, num_skills]) * common_skill_ability
    worker_skill_parameters = np.append(worker_skill_parameters, tem, axis=0)

    task_skill_need = {}
    task_community = {}
    for task in task_list:
        comunnity = np.random.choice(len(skill_dirichlet_parameters))
        task_community[task] = comunnity
        dirichlet_parameters = skill_dirichlet_parameters[comunnity]
        task_skill_need[task] = np.random.dirichlet(dirichlet_parameters)

    worker_ability = {}
    worker_community = {}
    for worker in worker_list:
        comunnity = np.random.choice(len(worker_skill_parameters))
        worker_community[worker] = comunnity
        worker_ability[worker] = worker_skill_parameters[comunnity]

    responses = {}
    truth = {}
    for task in task_list:
        responses[task] = {}
        truth[task] = random.choice(label_domain)
        for i in range(num_redundancy):
            worker = random.choice(worker_list)
            probability = np.dot(task_skill_need[task],worker_ability[worker])
            probability2 = (1-probability)/(num_domain-1)

            # simapling as the probability
            labels = [truth[task]]
            prob = [probability]
            for label in label_domain:
                if label != truth[task]:
                    labels.append(label)
                    prob.append(probability2)
            responses[task][worker] = np.random.choice(labels,p=prob)

    with open(truth_file, 'w', newline='', encoding='UTF-8') as f:
        writer = csv.writer(f)
        writer.writerow(['task_id', 'truth'])
        for task in truth:
            writer.writerow([task,truth[task]])

    with open(datafile, 'w', newline='', encoding='UTF-8') as f:
        writer = csv.writer(f)
        writer.writerow(['task', 'worker','answer'])
        for task in responses:
            for worker in responses[task]:
                writer.writerow([task,worker,responses[task][worker]])

    with open(task_community_file, 'w', newline='', encoding='UTF-8') as f:
        writer = csv.writer(f)
        writer.writerow(['task_id', 'task_community'])
        for task in task_community:
            writer.writerow([task, task_community[task]])

    with open(worker_community_file, 'w', newline='', encoding='UTF-8') as f:
        writer = csv.writer(f)
        writer.writerow(['worker_id', 'worker_community'])
        for worker in worker_community:
            writer.writerow([worker,worker_community[worker]])
    return datafile, truth_file, task_community_file, worker_community_file
