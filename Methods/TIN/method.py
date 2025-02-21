import csv

import math
import random


class TIN(object):

    def __init__(self, datafile, **kwargs):
        # settings
        self.theta_dir_prior = 2
        self.pi_dir_prior = 2
        self.dir_prior_multiplier = 10
        self.density = 5
        self.tol = 1e-6
        self.max_iteration = 300
        self.init_quality = 0.7

        # change settings
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])

        # initialise datafile
        setattr(self, 'datafile', datafile)
        t2wl, w2tl, is_labeled, task_set, worker_set, label_set, gross_labels = self.get_info_data()
        self.t2wl = t2wl
        self.w2tl = w2tl
        self.is_labeled = is_labeled
        self.task_set = task_set
        self.num_tasks = len(task_set)
        self.worker_set = worker_set
        self.num_workers = len(worker_set)
        self.label_set = label_set
        self.num_labels = len(label_set)
        self.gross_labels = gross_labels

        # initialise nets
        self.init_nets()
        # initialise parameters
        self.init_parameters()

    def init_nets(self):
        t2c, w2c, task_community_set, worker_community_set = self.get_info_community()
        self.t2c = t2c
        self.w2c = w2c
        self.task_community_set = task_community_set
        self.num_task_communities = len(task_community_set)
        self.worker_community_set = worker_community_set
        self.num_worker_communities = len(worker_community_set)

        probability_truth2label = self.get_info_label_relation()
        self.probability_truth2label = probability_truth2label

    def init_parameters(self):
        max_task_patterns = self.num_tasks * self.num_labels ** 2 / (self.num_tasks + self.num_labels ** 2)
        max_worker_patterns = self.num_workers * self.num_labels ** 2 / (self.num_workers + self.num_labels ** 2)
        self.num_task_patterns = math.ceil(
            min(self.gross_labels / (self.num_labels ** 2 + self.num_task_communities) / self.density, self.num_tasks,
                max_task_patterns))
        self.num_worker_patterns = math.ceil(
            min(self.gross_labels / (self.num_labels ** 2 + self.num_worker_communities) / self.density,
                self.num_workers, max_worker_patterns))

        self.task_pattern_set = set(range(self.num_task_patterns))
        self.worker_pattern_set = set(range(self.num_worker_patterns))

        # self.num_task_patterns = 20
        # self.num_worker_patterns = 20

        # i2g: ground truth of task i
        theta_i2g = {}
        for task in self.task_set:
            theta_i2g[task] = {}
            for label in self.label_set:
                theta_i2g[task][label] = 1 / self.num_labels
        self.theta_i2g = theta_i2g

        # tc2p: task community to its patterns
        theta_tc2p = {}
        for task_community in self.task_community_set:
            theta_tc2p[task_community] = [1 / self.num_task_patterns] * self.num_task_patterns
            for task_pattern in self.task_pattern_set:
                theta_tc2p[task_community][task_pattern] = random.random()

            count = sum(theta_tc2p[task_community])
            for task_pattern in self.task_pattern_set:
                theta_tc2p[task_community][task_pattern] /= count
        self.theta_tc2p = theta_tc2p

        # worker community to its patterns
        theta_wc2p = {}
        for worker_community in self.worker_community_set:
            theta_wc2p[worker_community] = [1 / self.num_worker_patterns] * self.num_worker_patterns
            for worker_pattern in self.worker_pattern_set:
                theta_wc2p[worker_community][worker_pattern] = random.random()

            count = sum(theta_wc2p[worker_community])
            for worker_pattern in self.worker_pattern_set:
                theta_wc2p[worker_community][worker_pattern] /= count
        self.theta_wc2p = theta_wc2p

        # confusion matrix of task pattern
        '''
        pi_tp = {}
        for task_pattern in self.task_pattern_set:
            quality = 0.5 + 0.5*task_pattern/self.num_task_patterns

            pi_tp[task_pattern] = {}
            for truth in self.label_set:
                pi_tp[task_pattern][truth] = {}
                for label in self.label_set:
                    if label == truth:
                        pi_tp[task_pattern][truth][label] = quality
                    else:
                        pi_tp[task_pattern][truth][label] = (1-quality)/(self.num_labels-1)

        self.pi_tp = pi_tp
        '''
        pi_tp = {}
        for task_pattern in self.task_pattern_set:
            pi_tp[task_pattern] = {}
            for truth in self.label_set:
                pi_tp[task_pattern][truth] = {}
                count = 0
                for label in self.label_set:
                    tem = random.random()
                    pi_tp[task_pattern][truth][label] = tem
                    count += tem
                # normalization
                for label in self.label_set:
                    pi_tp[task_pattern][truth][label] /= count

        self.pi_tp = pi_tp
        # confusion matrix of worker pattern
        '''
        pi_wp = {}
        for worker_pattern in self.worker_pattern_set:
            quality = 0.5 + 0.5*worker_pattern/self.num_worker_patterns

            pi_wp[worker_pattern] = {}
            for truth in self.label_set:
                pi_wp[worker_pattern][truth] = {}
                for label in self.label_set:
                    if label == truth:
                        pi_wp[worker_pattern][truth][label] = quality
                    else:
                        pi_wp[worker_pattern][truth][label] = (1-quality)/(self.num_labels-1)

        self.pi_wp = pi_wp
        '''
        pi_wp = {}
        for worker_pattern in self.worker_pattern_set:
            pi_wp[worker_pattern] = {}
            for truth in self.label_set:
                pi_wp[worker_pattern][truth] = {}
                count = 0
                for label in self.label_set:
                    tem = random.random()
                    pi_wp[worker_pattern][truth][label] = tem
                    count = + tem

                # normalization
                for label in self.label_set:
                    pi_wp[worker_pattern][truth][label] /= count
        self.pi_wp = pi_wp

        # confusion matrix of label
        self.pi_l = self.probability_truth2label

    def setting(self, **kwargs):
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])

    def f_func(self, task, worker, truth, task_pattern, worker_pattern):
        label = self.t2wl[task][worker]
        task_community = self.t2c[task]
        worker_community = self.w2c[worker]

        return self.theta_i2g[task][truth] * self.theta_tc2p[task_community][task_pattern] * \
               self.theta_wc2p[worker_community][worker_pattern] * self.pi_tp[task_pattern][truth][label] \
               * self.pi_wp[worker_pattern][truth][label] * self.pi_l[truth][label]

    # E-step
    def update_h(self):
        h = {}
        count = {}

        for task in self.task_set:
            h[task] = {}
            count[task] = {}

            for worker in self.t2wl[task]:
                h[task][worker] = {}
                count[task][worker] = 0

                for truth in self.label_set:
                    h[task][worker][truth] = {}
                    for task_pattern in self.task_pattern_set:
                        h[task][worker][truth][task_pattern] = {}
                        for worker_pattern in self.worker_pattern_set:
                            tem = self.f_func(task, worker, truth, task_pattern, worker_pattern)
                            h[task][worker][truth][task_pattern][worker_pattern] = tem
                            count[task][worker] += tem

        # normalization
        for task in self.task_set:
            for worker in self.t2wl[task]:
                for truth in self.label_set:
                    for task_pattern in self.task_pattern_set:
                        for worker_pattern in self.worker_pattern_set:
                            h[task][worker][truth][task_pattern][worker_pattern] /= count[task][worker]

        self.h = h


    # M-step
    def update_parameters(self):
        theta_i2g = {}
        count_theta_i2g = {}

        theta_tc2p = {}
        count_theta_tc2p = {}

        theta_wc2p = {}
        count_theta_wc2p = {}

        pi_tp = {}
        count_pi_tp = {}

        pi_wp = {}
        count_pi_wp = {}

        pi_l = {}
        count_pi_l = {}

        # initialization
        for task in self.task_set:
            theta_i2g[task] = {}
            count_theta_i2g[task] = 0

            task_community = self.t2c[task]
            theta_tc2p[task_community] = {}
            count_theta_tc2p[task_community] = 0
            for task_pattern in self.task_pattern_set:
                theta_tc2p[task_community][task_pattern] = 0

                pi_tp[task_pattern] = {}
                count_pi_tp[task_pattern] = {}
                for worker in self.t2wl[task]:
                    worker_community = self.w2c[worker]

                    theta_wc2p[worker_community] = {}
                    count_theta_wc2p[worker_community] = 0
                    for worker_pattern in self.worker_pattern_set:
                        theta_wc2p[worker_community][worker_pattern] = 0

                        pi_wp[worker_pattern] = {}
                        count_pi_wp[worker_pattern] = {}
                        for truth in self.label_set:
                            theta_i2g[task][truth] = 0

                            pi_l[truth] = {}
                            count_pi_l[truth] = 0

                            pi_tp[task_pattern][truth] = {}
                            count_pi_tp[task_pattern][truth] = 0

                            pi_wp[worker_pattern][truth] = {}
                            count_pi_wp[worker_pattern][truth] = 0
                            for label in self.label_set:
                                pi_l[truth][label] = 0

                                pi_tp[task_pattern][truth][label] = 0

                                pi_wp[worker_pattern][truth][label] = 0

        # computation
        for task in self.task_set:
            task_community = self.t2c[task]
            for task_pattern in self.task_pattern_set:
                for worker in self.t2wl[task]:
                    worker_community = self.w2c[worker]
                    for worker_pattern in self.worker_pattern_set:
                        for truth in self.label_set:
                            label = self.t2wl[task][worker]

                            tem = self.h[task][worker][truth][task_pattern][worker_pattern]

                            theta_i2g[task][truth] += tem
                            count_theta_i2g[task] += tem

                            theta_tc2p[task_community][task_pattern] += tem
                            count_theta_tc2p[task_community] += tem

                            theta_wc2p[worker_community][worker_pattern] += tem
                            count_theta_wc2p[worker_community] += tem

                            pi_tp[task_pattern][truth][label] += tem
                            count_pi_tp[task_pattern][truth] += tem

                            pi_wp[worker_pattern][truth][label] += tem
                            count_pi_wp[worker_pattern][truth] += tem

                            pi_l[truth][label] += tem
                            count_pi_l[truth] += tem

        # regularization
        for task in self.task_set:
            for truth in self.label_set:
                theta_i2g[task][truth] += self.theta_dir_prior - 1
                count_theta_i2g[task] += self.theta_dir_prior - 1

        for task_community in self.task_community_set:
            for task_pattern in self.task_pattern_set:
                theta_tc2p[task_community][task_pattern] += self.theta_dir_prior - 1
                count_theta_tc2p[task_community] += self.theta_dir_prior - 1

        for worker_community in self.worker_community_set:
            for worker_pattern in self.worker_pattern_set:
                theta_wc2p[worker_community][worker_pattern] += self.theta_dir_prior - 1
                count_theta_wc2p[worker_community] += self.theta_dir_prior - 1

        for task_pattern in self.task_pattern_set:
            for truth in self.label_set:
                for label in self.label_set:
                    pi_tp[task_pattern][truth][label] += self.pi_dir_prior - 1
                    count_pi_tp[task_pattern][truth] += self.pi_dir_prior - 1

        for worker_pattern in self.worker_pattern_set:
            for truth in self.label_set:
                for label in self.label_set:
                    pi_wp[worker_pattern][truth][label] += self.pi_dir_prior - 1
                    count_pi_wp[worker_pattern][truth] += self.pi_dir_prior - 1

        for truth in self.label_set:
            for label in self.label_set:
                pi_l[truth][label] += self.dir_prior_multiplier * self.probability_truth2label[truth][label]
                count_pi_l[truth] += self.dir_prior_multiplier * self.probability_truth2label[truth][label]

        # normalization of theta_g
        for task in self.task_set:
            for truth in self.label_set:
                theta_i2g[task][truth] /= count_theta_i2g[task]

        # normalization of theta_tc2p
        for task_community in self.task_community_set:
            for task_pattern in self.task_pattern_set:
                theta_tc2p[task_community][task_pattern] /= count_theta_tc2p[task_community]

        # normalization of theta_wc2p
        for worker_community in self.worker_community_set:
            for worker_pattern in self.worker_pattern_set:
                theta_wc2p[worker_community][worker_pattern] /= count_theta_wc2p[worker_community]

        # normalization of pi_tp
        for task_pattern in self.task_pattern_set:
            for truth in self.label_set:
                for label in self.label_set:
                    pi_tp[task_pattern][truth][label] /= count_pi_tp[task_pattern][truth]

        # normalization of pi_wp
        for worker_pattern in self.worker_pattern_set:
            for truth in self.label_set:
                for label in self.label_set:
                    pi_wp[worker_pattern][truth][label] /= count_pi_wp[worker_pattern][truth]

        # normalization of pi_l
        for truth in self.label_set:
            for label in self.label_set:
                pi_l[truth][label] /= count_pi_l[truth]

        self.theta_i2g = theta_i2g
        self.theta_tc2p = theta_tc2p
        self.theta_wc2p = theta_wc2p
        self.pi_tp = pi_tp
        self.pi_wp = pi_wp
        self.pi_l = pi_l

    # likelihood
    def get_likelihood(self):
        likelihood = 0

        for task in self.task_set:
            for worker in self.t2wl[task]:
                tem = 0
                for truth in self.label_set:
                    for task_pattern in self.task_pattern_set:
                        for worker_pattern in self.worker_pattern_set:
                            tem += self.f_func(task, worker, truth, task_pattern, worker_pattern)

                likelihood += math.log(tem)

        return likelihood

    def get_aux_func(self):
        aux = 0

        for task in self.task_set:
            for worker in self.t2wl[task]:
                for truth in self.label_set:
                    for task_pattern in self.task_pattern_set:
                        for worker_pattern in self.worker_pattern_set:
                            h = self.h[task][worker][truth][task_pattern][worker_pattern]
                            f = self.f_func(task, worker, truth, task_pattern, worker_pattern)
                            aux += h * math.log(f / h)

        # prior of theta_g
        for task in self.task_set:
            for truth in self.label_set:
                aux += (self.theta_dir_prior - 1) * math.log(self.theta_i2g[task][truth])

        # prior of theta_tc2p
        for task_community in self.task_community_set:
            for task_pattern in self.task_pattern_set:
                aux += (self.theta_dir_prior - 1) * math.log(self.theta_tc2p[task_community][task_pattern])

        # prior of theta_wc2p
        for worker_community in self.worker_community_set:
            for worker_pattern in self.worker_pattern_set:
                aux += (self.theta_dir_prior - 1) * math.log(self.theta_wc2p[worker_community][worker_pattern])

        # prior of pi_tp
        for task_pattern in self.task_pattern_set:
            for truth in self.label_set:
                for label in self.label_set:
                    aux += (self.pi_dir_prior - 1) * math.log(self.pi_tp[task_pattern][truth][label])

        # normalization of pi_wp
        for worker_pattern in self.worker_pattern_set:
            for truth in self.label_set:
                for label in self.label_set:
                    aux += (self.pi_dir_prior - 1) * math.log(self.pi_wp[worker_pattern][truth][label])

        # normalization of pi_l
        for truth in self.label_set:
            for label in self.label_set:
                aux += (self.pi_dir_prior - 1) * math.log(self.pi_l[truth][label])

        return aux

    def run(self):
        aux_m = float('-inf')

        for _ in range(self.max_iteration):
            theta_i2g = self.theta_i2g
            # E-step
            self.update_h()
            # print('h function has updated successfully!')

            aux_e = self.get_aux_func()
            # print('the value of aux func is %e for E-step' % aux_e)
            inc = aux_e - aux_m
            print('the increment of aux func is %e for E-step' % inc)

            # likelihood = self.get_likelihood()
            # print('The likelihood is %e' % likelihood)

            # M-step
            self.update_parameters()
            # print('Parameters have updated successfully!')

            aux_m = self.get_aux_func()
            # print('the value of aux func is %e for M-step' % aux_m)
            inc = aux_m - aux_e
            print('the increment of aux func is %e for M-step' % inc)

            error = self.get_diff_theta_i2g(theta_i2g, self.theta_i2g)
            # print('The error is %e' % error)
            if error < self.tol:
                break

        t2a = {}
        for task in self.task_set:
            t2a[task] = random.choice(list(self.label_set))
            for truth in self.label_set:
                if self.theta_i2g[task][truth] > self.theta_i2g[task][t2a[task]]:
                    t2a[task] = truth

        self.t2a = t2a

    def get_diff_theta_i2g(self, theta_i2g, new_theta_i2g):
        diff = 0
        for task in self.task_set:
            for truth in self.label_set:
                diff += abs(theta_i2g[task][truth] - new_theta_i2g[task][truth])

        return diff

    def get_info_data(self):
        if not hasattr(self, 'datafile'):
            raise BaseException('There is no datafile!')

        t2wl = {}
        w2tl = {}
        is_labeled = {}
        task_set = set()
        worker_set = set()
        label_set = set()
        gross_labels = 0

        f = open(self.datafile, 'r')
        reader = csv.reader(f)
        next(reader)

        for line in reader:
            task, worker, label = line

            if task not in t2wl:
                t2wl[task] = {}
            t2wl[task][worker] = label

            if worker not in w2tl:
                w2tl[worker] = {}
            w2tl[worker][task] = label

            if task not in task_set:
                task_set.add(task)

            if worker not in worker_set:
                worker_set.add(worker)

            if label not in label_set:
                label_set.add(label)

            for task in task_set:
                is_labeled[task] = {}
                for worker in worker_set:
                    if worker in t2wl[task]:
                        is_labeled[task][worker] = True
                        gross_labels += 1
                    else:
                        is_labeled[task][worker] = False

        return t2wl, w2tl, is_labeled, task_set, worker_set, label_set, gross_labels

    def get_info_community(self):

        t2c = {}
        w2c = {}
        task_community_set = set()
        worker_community_set = set()

        # task community
        for task in self.task_set:
            t2c[task] = task

        if hasattr(self, 'task_community_file'):
            f = open(self.task_community_file, 'r')
            reader = csv.reader(f)
            next(reader)

            for line in reader:
                task, community = line
                t2c[task] = community

        for task in t2c:
            community = t2c[task]
            task_community_set.add(community)

        # worker community
        for worker in self.worker_set:
            w2c[worker] = worker

        if hasattr(self, 'worker_community_file'):
            f = open(self.worker_community_file, 'r')
            reader = csv.reader(f)
            next(reader)

            for line in reader:
                worker, community = line
                w2c[worker] = community

        for worker in w2c:
            community = w2c[worker]
            worker_community_set.add(community)

        self.t2c = t2c
        self.w2c = w2c
        self.task_community_set = task_community_set
        self.worker_community_set = worker_community_set

        return t2c, w2c, task_community_set, worker_community_set

    def get_info_label_relation(self):
        probability_truth2label = {}

        for truth in self.label_set:
            probability_truth2label[truth] = {}
            for label in self.label_set:
                probability_truth2label[truth][label] = self.init_quality if truth == label else (
                                                                                                         1 - self.init_quality) / (
                                                                                                         self.num_labels - 1)

        if hasattr(self, 'label_relation_file'):
            f = open(self.label_relation_file, 'r')
            reader = csv.reader(f)
            next(reader)

            for line in reader:
                truth, label, prob = line
                probability_truth2label[truth][label] = prob

            # normalization
            for truth in self.label_set:
                count = 0

                for label in self.label_set:
                    count += probability_truth2label[truth][label]

                for label in self.label_set:
                    probability_truth2label[truth][label] /= count

        self.probability_truth2label = probability_truth2label

        return probability_truth2label

    def get_accuracy(self):
        if not hasattr(self, 'truth_file'):
            raise BaseException('There is no truth file!')
        if not hasattr(self, 't2a'):
            raise BaseException('There is no aggregated answers!')

        count = []
        f = open(self.truth_file, 'r')
        reader = csv.reader(f)
        next(reader)

        for line in reader:
            task, truth = line
            if task not in self.t2a:
                print('task %s in not found in the list of answers' % task)
            elif self.t2a[task] == truth:
                count.append(1)
            else:
                count.append(0)

        return sum(count) / len(count)


if __name__ == '__main__':
    # datafile = r'D:\python_workspace\TIN\datasets\d_Duck_Identification\answer.csv'
    # truth_file=r'D:\python_workspace\TIN\datasets\d_Duck_Identification\truth.csv'

    datafile = r'D:\python_workspace\TIN\datasets\d_jn-product\answer.csv'
    truth_file = r'D:\python_workspace\TIN\datasets\d_jn-product\truth.csv'

    # datafile = r'D:\python_workspace\TIN\datasets\f201_Emotion_FULL\answer.csv'
    # truth_file = r'D:\python_workspace\TIN\datasets\f201_Emotion_FULL\truth.csv'

    tin = TIN(datafile, truth_file=truth_file)

    print('object has created successfully!')
    tin.run()
    accuracy = tin.get_accuracy()
    print('The accuracy of TIN is %f' % accuracy)
