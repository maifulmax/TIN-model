import csv
import math
import random
import time


class TIN(object):
    __slots__ = (
        'theta_dir_prior', 'pi_dir_prior', 'dir_prior_multiplier', 'density', 'max_patterns', 'tol', 'max_iteration',
        'init_quality', 't2wl', 't2l', 'w2tl', 'w2l', 'l2tw', 'task_set', 'num_tasks', 'worker_set', 'num_workers',
        'label_set',
        'num_labels', 'gross_labels', 't2c', 'w2c', 'c2t', 'tc2l', 'c2w', 'wc2l', 'task_community_set',
        'num_task_communities',
        'worker_community_set', 'num_worker_communities', 'probability_truth2label', 'num_task_patterns',
        'num_worker_patterns', 'task_pattern_set', 'worker_pattern_set', 'theta_i2g', 'theta_tc2p', 'theta_wc2p',
        'pi_tp', 'pi_wp', 'pi_l', 'phi_ig', 'phi_gkdr', 'phi_gkcq', 't2a', 'datafile', 'truth_file',
        'task_community_file', 'worker_community_file', 'label_relation_file')

    def __init__(self, datafile, **kwargs):
        # settings
        self.theta_dir_prior = 2 + 1e-4
        self.pi_dir_prior = 2 + 1e-4
        # self.dir_prior_multiplier = 0
        self.density = 5
        self.max_patterns = 10
        self.tol = 1e-6
        self.max_iteration = 100
        self.init_quality = 0.7

        setattr(self, 'datafile', datafile)
        # change settings
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])

        # initialise datafile
        t2wl, t2l, w2tl, w2l, l2tw, task_set, worker_set, label_set, gross_labels = self.get_info_data()
        self.t2wl = t2wl
        self.t2l = t2l
        self.w2tl = w2tl
        self.w2l = w2l
        self.l2tw = l2tw
        self.task_set = task_set
        self.num_tasks = len(task_set)
        self.worker_set = worker_set
        self.num_workers = len(worker_set)
        self.label_set = label_set
        self.num_labels = len(label_set)
        self.gross_labels = gross_labels
        # print('getting info of data is complete')

        # initialise nets
        self.init_nets()
        # print('init nets is complete')

        # initialise parameters
        self.init_parameters()
        # print('init parameters is complete')

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
        num_task_patterns = math.ceil(
            min(self.gross_labels / (self.num_labels ** 2 + self.num_task_communities) / self.density, self.num_tasks,
                max_task_patterns, self.max_patterns))
        num_worker_patterns = math.ceil(
            min(self.gross_labels / (self.num_labels ** 2 + self.num_worker_communities) / self.density,
                self.num_workers, max_worker_patterns, self.max_patterns))

        self.num_task_patterns = max(num_task_patterns, 2)
        self.num_worker_patterns = max(num_worker_patterns, 2)

        self.task_pattern_set = set(range(self.num_task_patterns))
        self.worker_pattern_set = set(range(self.num_worker_patterns))

        # i2g: ground truth of task i
        theta_i2g = {}
        for task in self.task_set:
            theta_i2g[task] = {}
            for label in self.label_set:
                theta_i2g[task][label] = 1 / self.num_labels
        self.theta_i2g = theta_i2g

        # # tc2p: task community to its patterns
        # theta_tc2p = {}
        # for task_community in self.task_community_set:
        #     theta_tc2p[task_community] = {}
        #
        #     for task_pattern in self.task_pattern_set:
        #         theta_tc2p[task_community][task_pattern] = 1 / self.num_task_patterns
        #
        # self.theta_tc2p = theta_tc2p

        # worker community to its patterns
        theta_wc2p = {}
        for worker_community in self.worker_community_set:
            theta_wc2p[worker_community] = {}

            for worker_pattern in self.worker_pattern_set:
                theta_wc2p[worker_community][worker_pattern] = 1 / self.num_worker_patterns

        self.theta_wc2p = theta_wc2p

        # # confusion matrix of task pattern
        # pi_tp = {}
        # for task_pattern in self.task_pattern_set:
        #     quality = 0.5 + 0.5 * task_pattern / self.num_task_patterns
        #
        #     pi_tp[task_pattern] = {}
        #     for truth in self.label_set:
        #         pi_tp[task_pattern][truth] = {}
        #         for label in self.label_set:
        #             if label == truth:
        #                 pi_tp[task_pattern][truth][label] = quality
        #             else:
        #                 pi_tp[task_pattern][truth][label] = (1 - quality) / (self.num_labels - 1)
        #
        # self.pi_tp = pi_tp

        # pi_tp = {}
        # for task_pattern in self.task_pattern_set:
        #     pi_tp[task_pattern] = {}
        #     for truth in self.label_set:
        #         pi_tp[task_pattern][truth] = {}
        #         count = 0
        #         for label in self.label_set:
        #             tem = random.random()
        #             pi_tp[task_pattern][truth][label] = tem
        #             count += tem
        #         # normalization
        #         for label in self.label_set:
        #             pi_tp[task_pattern][truth][label] /= count
        #
        # self.pi_tp = pi_tp

        # confusion matrix of worker pattern
        pi_wp = {}
        for worker_pattern in self.worker_pattern_set:
            quality = 0.5 + 0.5 * worker_pattern / self.num_worker_patterns

            pi_wp[worker_pattern] = {}
            for truth in self.label_set:
                pi_wp[worker_pattern][truth] = {}
                for label in self.label_set:
                    if label == truth:
                        pi_wp[worker_pattern][truth][label] = quality
                    else:
                        pi_wp[worker_pattern][truth][label] = (1 - quality) / (self.num_labels - 1)

        self.pi_wp = pi_wp

        # pi_wp = {}
        # for worker_pattern in self.worker_pattern_set:
        #     pi_wp[worker_pattern] = {}
        #     for truth in self.label_set:
        #         pi_wp[worker_pattern][truth] = {}
        #         count = 0
        #         for label in self.label_set:
        #             tem = random.random()
        #             pi_wp[worker_pattern][truth][label] = tem
        #             count = + tem
        #
        #         # normalization
        #         for label in self.label_set:
        #             pi_wp[worker_pattern][truth][label] /= count
        # self.pi_wp = pi_wp

        # # confusion matrix of label
        # self.pi_l = self.probability_truth2label

    def setting(self, **kwargs):
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])

    # E-step
    def update_phi_ig(self):
        # t1 = time.time()
        phi_ig = {}
        for task in self.task_set:
            phi_ig[task] = {}
            task_community = self.t2c[task]

            count = 0
            for truth in self.label_set:
                prod_g = self.theta_i2g[task][truth]

                for worker in self.t2wl[task]:
                    label = self.t2wl[task][worker]
                    worker_community = self.w2c[worker]

                    # sum_tp = 0
                    # for task_pattern in self.task_pattern_set:
                    #     sum_tp += self.theta_tc2p[task_community][task_pattern] * self.pi_tp[task_pattern][truth][label]

                    sum_wp = 0
                    for worker_pattern in self.worker_pattern_set:
                        sum_wp += self.theta_wc2p[worker_community][worker_pattern] * self.pi_wp[worker_pattern][truth][
                            label]

                    prod_g *= sum_wp
                phi_ig[task][truth] = prod_g
                count += prod_g

            # normalization
            for truth in self.label_set:
                phi_ig[task][truth] /= count

        self.phi_ig = phi_ig

        # t2 = time.time()
        # print('The time cost in update phi_ig is %e' %(t2-t1))

        return

    def update_phi_gkdr(self):
        # t1 = time.time()
        phi_gkdr = {}

        for task_community in self.task_community_set:
            phi_gkdr[task_community] = {}

            for truth in self.label_set:
                phi_gkdr[task_community][truth] = {}

                for label in self.tc2l[task_community]:
                    phi_gkdr[task_community][truth][label] = {}

                    count = 0
                    for task_pattern in self.task_pattern_set:
                        tem = self.theta_tc2p[task_community][task_pattern] * self.pi_tp[task_pattern][truth][label]
                        phi_gkdr[task_community][truth][label][task_pattern] = tem
                        count += tem

                    # normalization
                    for task_pattern in self.task_pattern_set:
                        phi_gkdr[task_community][truth][label][task_pattern] /= count

        self.phi_gkdr = phi_gkdr
        # t2 = time.time()
        # print('The time cost in update phi_gkdr is %e' % (t2 - t1))
        return

    def update_phi_gkcq(self):
        # t1 = time.time()
        phi_gkcq = {}

        for worker_community in self.worker_community_set:
            phi_gkcq[worker_community] = {}

            for truth in self.label_set:
                phi_gkcq[worker_community][truth] = {}

                for label in self.wc2l[worker_community]:
                    phi_gkcq[worker_community][truth][label] = {}

                    count = 0
                    for worker_pattern in self.worker_pattern_set:
                        tem = self.theta_wc2p[worker_community][worker_pattern] * self.pi_wp[worker_pattern][truth][
                            label]
                        phi_gkcq[worker_community][truth][label][worker_pattern] = tem
                        count += tem

                    # normalization
                    for worker_pattern in self.worker_pattern_set:
                        phi_gkcq[worker_community][truth][label][worker_pattern] /= count

        self.phi_gkcq = phi_gkcq
        # t2 = time.time()
        # print('The time cost in update phi_gkcq is %e' % (t2 - t1))
        return

    # M-step
    def update_parameters(self):
        theta_i2g = {}
        theta_tc2p = {}
        theta_wc2p = {}
        pi_tp = {}
        pi_wp = {}
        pi_l = {}

        # compute theta_i2g
        # t1 = time.time()
        for task in self.task_set:
            theta_i2g[task] = {}

            count = 0
            for truth in self.label_set:
                theta_i2g[task][truth] = self.phi_ig[task][truth] + self.theta_dir_prior - 1
                count += theta_i2g[task][truth]

            for truth in self.label_set:
                theta_i2g[task][truth] /= count

        # t2 = time.time()
        # print('The time cost in update theta_i2g is %e' % (t2 - t1))

        # # compute theta_tc2p
        # t1 = time.time()
        # for task_community in self.task_community_set:
        #     theta_tc2p[task_community] = {}
        #
        #     tem = 0
        #     for task_pattern in self.task_pattern_set:
        #         count = 0
        #
        #         for task in self.c2t[task_community]:
        #             for truth in self.label_set:
        #                 for worker in self.t2wl[task]:
        #                     label = self.t2wl[task][worker]
        #                     count += self.phi_ig[task][truth] * self.phi_gkdr[task_community][truth][label][
        #                         task_pattern]
        #         count += self.theta_dir_prior - 1
        #         theta_tc2p[task_community][task_pattern] = count
        #         tem += count
        #
        #     for task_pattern in self.task_pattern_set:
        #         theta_tc2p[task_community][task_pattern] /= tem
        #
        # t2 = time.time()
        # print('The time cost in update theta_tc2p is %e' % (t2 - t1))

        # compute theta_wc2p
        # t1 = time.time()
        for worker_community in self.worker_community_set:
            theta_wc2p[worker_community] = {}

            tem = 0
            for worker_pattern in self.worker_pattern_set:
                count = 0

                for worker in self.c2w[worker_community]:
                    for task in self.w2tl[worker]:
                        label = self.w2tl[worker][task]

                        for truth in self.label_set:
                            count += self.phi_ig[task][truth] * self.phi_gkcq[worker_community][truth][label][
                                worker_pattern]
                count += self.theta_dir_prior - 1
                theta_wc2p[worker_community][worker_pattern] = count
                tem += count

            for worker_pattern in self.worker_pattern_set:
                theta_wc2p[worker_community][worker_pattern] /= tem

        # t2 = time.time()
        # print('The time cost in update theta_wc2p is %e' % (t2 - t1))

        # # compute pi_l
        # t1 = time.time()
        # for truth in self.label_set:
        #     pi_l[truth] = {}
        #
        #     tem = 0
        #     for label in self.label_set:
        #         count = 0
        #
        #         for task, worker in self.l2tw[label]:
        #             count += self.phi_ig[task][truth]
        #         count += self.dir_prior_multiplier * self.probability_truth2label[truth][label]
        #         pi_l[truth][label] = count
        #         tem += count
        #
        #     for label in self.label_set:
        #         pi_l[truth][label] /= tem
        #
        # t2 = time.time()
        # print('The time cost in update pi_l is %e' % (t2 - t1))
        #
        # # compute pi_tp
        # t1 = time.time()
        # for task_pattern in self.task_pattern_set:
        #     pi_tp[task_pattern] = {}
        #
        #     for truth in self.label_set:
        #         pi_tp[task_pattern][truth] = {}
        #
        #         tem = 0
        #         for label in self.label_set:
        #             count = 0
        #
        #             for task, worker in self.l2tw[label]:
        #                 task_community = self.t2c[task]
        #                 count += self.phi_ig[task][truth] * self.phi_gkdr[task_community][truth][label][task_pattern]
        #             count += self.pi_dir_prior - 1
        #             pi_tp[task_pattern][truth][label] = count
        #             tem += count
        #
        #         for label in self.label_set:
        #             pi_tp[task_pattern][truth][label] /= tem
        #
        # t2 = time.time()
        # print('The time cost in update pi_tp is %e' % (t2 - t1))

        # compute pi_wp
        # t1 = time.time()
        for worker_pattern in self.worker_pattern_set:
            pi_wp[worker_pattern] = {}

            for truth in self.label_set:
                pi_wp[worker_pattern][truth] = {}

                tem = 0
                for label in self.label_set:
                    count = 0

                    for task, worker in self.l2tw[label]:
                        worker_community = self.w2c[worker]

                        count += self.phi_ig[task][truth] * self.phi_gkcq[worker_community][truth][label][
                            worker_pattern]
                    count += self.pi_dir_prior - 1
                    pi_wp[worker_pattern][truth][label] = count
                    tem += count

                for label in self.label_set:
                    pi_wp[worker_pattern][truth][label] /= tem

        # t2 = time.time()
        # print('The time cost in update pi_wp is %e' % (t2 - t1))

        self.theta_i2g = theta_i2g
        # self.theta_tc2p = theta_tc2p
        self.theta_wc2p = theta_wc2p
        # self.pi_tp = pi_tp
        self.pi_wp = pi_wp
        # self.pi_l = pi_l

    # likelihood
    def get_likelihood(self):
        likelihood = 0
        for task in self.task_set:
            task_community = self.t2c[task]

            sum_g = 0
            for truth in self.label_set:

                prod_j = self.theta_i2g[task][truth]
                for worker in self.t2wl[task]:
                    label = self.t2wl[task][worker]

                    sum_pi_j = 0
                    worker_community = self.w2c[worker]
                    for worker_pattern in self.worker_pattern_set:
                        sum_pi_j += self.theta_wc2p[worker_community][worker_pattern] * \
                                    self.pi_wp[worker_pattern][truth][label]

                    prod_j *= sum_pi_j

                sum_g += prod_j

            likelihood += math.log(sum_g)

        # prior of theta_g
        for task in self.task_set:
            for truth in self.label_set:
                likelihood += (self.theta_dir_prior - 1) * math.log(self.theta_i2g[task][truth])

        # # prior of theta_tc2p
        # for task_community in self.task_community_set:
        #     for task_pattern in self.task_pattern_set:
        #         likelihood += (self.theta_dir_prior - 1) * math.log(self.theta_tc2p[task_community][task_pattern])

        # prior of theta_wc2p
        for worker_community in self.worker_community_set:
            for worker_pattern in self.worker_pattern_set:
                likelihood += (self.theta_dir_prior - 1) * math.log(self.theta_wc2p[worker_community][worker_pattern])

        # # prior of pi_tp
        # for task_pattern in self.task_pattern_set:
        #     for truth in self.label_set:
        #         for label in self.label_set:
        #             likelihood += (self.pi_dir_prior - 1) * math.log(self.pi_tp[task_pattern][truth][label])

        # normalization of pi_wp
        for worker_pattern in self.worker_pattern_set:
            for truth in self.label_set:
                for label in self.label_set:
                    likelihood += (self.pi_dir_prior - 1) * math.log(self.pi_wp[worker_pattern][truth][label])

        # # normalization of pi_l
        # for truth in self.label_set:
        #     for label in self.label_set:
        #         likelihood += (self.pi_dir_prior - 1) * math.log(self.pi_l[truth][label])

        return likelihood

    def get_aux_func(self):
        aux = 0

        for task in self.task_set:
            for truth in self.label_set:
                if self.phi_ig[task][truth] == 0:
                    continue
                aux += self.phi_ig[task][truth] * math.log(self.theta_i2g[task][truth] / self.phi_ig[task][truth])

                for worker in self.t2wl[task]:
                    label = self.t2wl[task][worker]
                    # aux += self.phi_ig[task][truth] * math.log(self.pi_l[truth][label])

                    # task_community = self.t2c[task]
                    # for task_pattern in self.task_pattern_set:
                    #     aux += self.phi_ig[task][truth] * self.phi_gkdr[task_community][truth][label][
                    #         task_pattern] * math.log(
                    #         self.theta_tc2p[task_community][task_pattern] * self.pi_tp[task_pattern][truth][label] /
                    #         self.phi_gkdr[task_community][truth][label][task_pattern])

                    worker_community = self.w2c[worker]
                    for worker_pattern in self.worker_pattern_set:
                        aux += self.phi_ig[task][truth] * self.phi_gkcq[worker_community][truth][label][
                            worker_pattern] * math.log(
                            self.theta_wc2p[worker][worker_pattern] * self.pi_wp[worker_pattern][truth][label] /
                            self.phi_gkcq[worker_community][truth][label][worker_pattern])

        # prior of theta_g
        for task in self.task_set:
            for truth in self.label_set:
                aux += (self.theta_dir_prior - 1) * math.log(self.theta_i2g[task][truth])

        # prior of theta_wc2p
        for worker_community in self.worker_community_set:
            for worker_pattern in self.worker_pattern_set:
                aux += (self.theta_dir_prior - 1) * math.log(self.theta_wc2p[worker_community][worker_pattern])

        # normalization of pi_wp
        for worker_pattern in self.worker_pattern_set:
            for truth in self.label_set:
                for label in self.label_set:
                    aux += (self.pi_dir_prior - 1) * math.log(self.pi_wp[worker_pattern][truth][label])

        return aux

    def run(self):
        aux_m = float('-inf')

        for iter in range(self.max_iteration):
            # E-step
            self.update_phi_ig()
            # print('phi_ig is updated')
            # self.update_phi_gkdr()
            # print('phi_gkdr is updated')
            self.update_phi_gkcq()
            # print('phi_gkcq is updated')

            # aux_e = self.get_aux_func()
            # inc = (aux_e - aux_m) / (-aux_m)
            # print('the increment of aux func is %e for E-step' % inc)

            # likelihood = self.get_likelihood()
            # print('The likelihood is %e' % likelihood)

            # likelihood = self.get_likelihood()
            #
            # diff = aux_e - likelihood
            # print('aux_e - likelihood is %e' % diff)

            # M-step
            self.update_parameters()
            # print('parameters is updated')

            # aux_m = self.get_aux_func()
            # inc = (aux_m - aux_e)
            # print('the increment of aux func is %e for M-step' % inc)

            # if inc_percentage < self.tol:
            #     break

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
        t2l = {}
        w2tl = {}
        w2l = {}
        l2tw = {}
        task_set = set()
        worker_set = set()
        label_set = set()
        gross_labels = 0

        with open(self.datafile, 'r') as f:
            reader = csv.reader(f)
            next(reader)

            for line in reader:
                task, worker, label = line
                gross_labels += 1

                if task not in t2wl:
                    t2wl[task] = {}
                t2wl[task][worker] = label

                if task not in t2l:
                    t2l[task] = set()
                t2l[task].add(label)

                if worker not in w2tl:
                    w2tl[worker] = {}
                w2tl[worker][task] = label

                if worker not in w2l:
                    w2l[worker] = set()
                w2l[worker].add(label)

                if label not in l2tw:
                    l2tw[label] = set()
                l2tw[label].add((task, worker))

                if task not in task_set:
                    task_set.add(task)

                if worker not in worker_set:
                    worker_set.add(worker)

                if label not in label_set:
                    label_set.add(label)

        return t2wl, t2l, w2tl, w2l, l2tw, task_set, worker_set, label_set, gross_labels

    def get_info_community(self):

        t2c = {}
        w2c = {}
        c2t = {}
        c2w = {}
        tc2l = {}
        wc2l = {}
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

        # c2t task community to task
        for community in task_community_set:
            c2t[community] = set()

        for task in t2c:
            community = t2c[task]
            c2t[community].add(task)

        # c2w worker community to worker
        for community in worker_community_set:
            c2w[community] = set()

        for worker in w2c:
            community = w2c[worker]
            c2w[community].add(worker)

        # tc2l task community to label
        for task_community in task_community_set:
            tc2l[task_community] = set()
            for task in c2t[task_community]:
                for label in self.t2l[task]:
                    tc2l[task_community].add(label)
        # wc2l task community to label
        for worker_community in worker_community_set:
            wc2l[worker_community] = set()
            for worker in c2w[worker_community]:
                for label in self.w2l[worker]:
                    wc2l[worker_community].add(label)

        self.t2c = t2c
        self.w2c = w2c
        self.c2t = c2t
        self.c2w = c2w
        self.tc2l = tc2l
        self.wc2l = wc2l
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
                if truth not in probability_truth2label:
                    probability_truth2label[truth] = {}
                probability_truth2label[truth][label] = float(prob)

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
        with open(self.truth_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)

            for task, truth in reader:
                if task not in self.t2a:
                    print('task %s in not found in the list of answers' % task)
                elif self.t2a[task] == truth:
                    count.append(1)
                else:
                    count.append(0)

        return sum(count) / len(count)


if __name__ == '__main__':
    # datafile = r'./datasets/test/answer.csv'
    # truth_file = r'./datasets/test/truth.csv'

    datafile = r'./datasets/d_Duck_Identification/answer.csv'
    truth_file = r'./datasets/d_Duck_Identification/truth.csv'

    # datafile = r'./datasets/s4_Dog_data/answer.csv'
    # truth_file=r'./datasets/s4_Dog_data/truth.csv'

    # datafile = r'./datasets/BTSK/data_redun19.csv'
    # truth_file = r'./datasets/BTSK/truth.csv'
    # label_relation_file = r'./datasets/BTSK/label_relation.csv'

    # datafile = r'D:/python_workspace/TIN/datasets/d_jn_product/answer.csv'
    # truth_file = r'D:/python_workspace/TIN/datasets/d_jn_product/truth.csv'

    # datafile = r'D:\python_workspace/TIN/datasets/f201_Emotion_FULL/answer.csv'
    # truth_file = r'D:\python_workspace/TIN/datasets/f201_Emotion_FULL/truth.csv'

    # datafile = r'D:\python_workspace\TIN\datasets\MS\answer.csv'
    # truth_file = r'D:\python_workspace\TIN\datasets\MS\truth.csv'

    tin = TIN(datafile, truth_file=truth_file)

    print('object has created successfully!')
    tin.run()
    accuracy = tin.get_accuracy()
    print('The accuracy of TIN is %f' % accuracy)
