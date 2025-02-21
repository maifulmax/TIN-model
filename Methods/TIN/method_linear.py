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
        'pi_tp', 'pi_wp', 'pi_l', 'phi_ig', 'phi_dcgk', 'phi_dcgkr', 'phi_dcgkq', 't2a', 'datafile', 'truth_file',
        'task_community_file', 'worker_community_file', 'label_relation_file', 'beta_1', 'beta_2', 'beta_3', 'tcwc2l')

    def __init__(self, datafile, **kwargs):
        # settings
        self.theta_dir_prior = 2 + 1e-4
        self.pi_dir_prior = 2 + 1e-4
        self.dir_prior_multiplier = 1
        self.density = 5
        self.max_patterns = 10
        self.tol = 1e-6
        self.max_iteration = 100
        self.init_quality = 0.8
        self.beta_1 = 0.15
        self.beta_2 = 0.05
        self.beta_3 = 0.8

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
        # beta_2 = gross_labels/len(task_set)
        # beta_3 = gross_labels/len(worker_set)
        # self.beta_2 = 0.8*beta_2/(beta_2+beta_3)
        # self.beta_3 = 0.8*beta_3/(beta_2+beta_3)
        # print('getting info of data is complete')

        # initialise nets
        self.init_nets()
        # print('init nets is complete')

        # initialise parameters
        self.init_parameters()
        # print('init parameters is complete')

    def init_nets(self):
        self.init_info_community()

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

        # tc2p: task community to its patterns
        theta_tc2p = {}
        for task_community in self.task_community_set:
            theta_tc2p[task_community] = {}

            for task_pattern in self.task_pattern_set:
                theta_tc2p[task_community][task_pattern] = 1 / self.num_task_patterns

        self.theta_tc2p = theta_tc2p

        # worker community to its patterns
        theta_wc2p = {}
        for worker_community in self.worker_community_set:
            theta_wc2p[worker_community] = {}

            for worker_pattern in self.worker_pattern_set:
                theta_wc2p[worker_community][worker_pattern] = 1 / self.num_worker_patterns

        self.theta_wc2p = theta_wc2p

        # confusion matrix of task pattern
        pi_tp = {}
        for task_pattern in self.task_pattern_set:
            quality = 0.5 + 0.5 * task_pattern / self.num_task_patterns

            pi_tp[task_pattern] = {}
            for truth in self.label_set:
                pi_tp[task_pattern][truth] = {}
                for label in self.label_set:
                    if label == truth:
                        pi_tp[task_pattern][truth][label] = quality
                    else:
                        pi_tp[task_pattern][truth][label] = (1 - quality) / (self.num_labels - 1)

        self.pi_tp = pi_tp

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

        # confusion matrix of label
        self.pi_l = self.probability_truth2label

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

                    sum_tp = 0
                    for task_pattern in self.task_pattern_set:
                        sum_tp += self.theta_tc2p[task_community][task_pattern] * self.pi_tp[task_pattern][truth][label]

                    sum_wp = 0
                    for worker_pattern in self.worker_pattern_set:
                        sum_wp += self.theta_wc2p[worker_community][worker_pattern] * self.pi_wp[worker_pattern][truth][
                            label]

                    prod_g *= self.beta_1 * self.pi_l[truth][label] + self.beta_2 * sum_tp + self.beta_3 * sum_wp
                phi_ig[task][truth] = prod_g
                count += prod_g

            # normalization
            for truth in self.label_set:
                phi_ig[task][truth] /= count

        self.phi_ig = phi_ig

        # t2 = time.time()
        # print('The time cost in update phi_ig is %e' %(t2-t1))

        return

    def update_phi_gk(self):
        # t1 = time.time()
        phi_dcgk = {}
        phi_dcgkr = {}
        phi_dcgkq = {}

        for task_community in self.tcwc2l:
            phi_dcgk[task_community] = {}
            phi_dcgkr[task_community] = {}
            phi_dcgkq[task_community] = {}

            for worker_community in self.tcwc2l[task_community]:
                phi_dcgk[task_community][worker_community] = {}
                phi_dcgkr[task_community][worker_community] = {}
                phi_dcgkq[task_community][worker_community] = {}

                for truth in self.label_set:
                    phi_dcgk[task_community][worker_community][truth] = {}
                    phi_dcgkr[task_community][worker_community][truth] = {}
                    phi_dcgkq[task_community][worker_community][truth] = {}

                    for label in self.tcwc2l[task_community][worker_community]:
                        phi_dcgk[task_community][worker_community][truth][label] = self.beta_1 * self.pi_l[truth][label]
                        phi_dcgkr[task_community][worker_community][truth][label] = {}
                        phi_dcgkq[task_community][worker_community][truth][label] = {}

                        count = self.beta_1 * self.pi_l[truth][label]

                        for task_pattern in self.task_pattern_set:
                            tem = self.beta_2 * self.theta_tc2p[task_community][task_pattern] * \
                                  self.pi_tp[task_pattern][truth][label]
                            phi_dcgkr[task_community][worker_community][truth][label][task_pattern] = tem
                            count += tem

                        for worker_pattern in self.worker_pattern_set:
                            tem = self.beta_3 * self.theta_wc2p[worker_community][worker_pattern] * \
                                  self.pi_wp[worker_pattern][truth][label]
                            phi_dcgkq[task_community][worker_community][truth][label][worker_pattern] = tem
                            count += tem

                        phi_dcgk[task_community][worker_community][truth][label] /= count

                        for task_pattern in self.task_pattern_set:
                            phi_dcgkr[task_community][worker_community][truth][label][task_pattern] /= count

                        for worker_pattern in self.worker_pattern_set:
                            phi_dcgkq[task_community][worker_community][truth][label][worker_pattern] /= count

        self.phi_dcgk = phi_dcgk
        self.phi_dcgkr = phi_dcgkr
        self.phi_dcgkq = phi_dcgkq

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

        # compute theta_tc2p
        # t1 = time.time()
        for task_community in self.task_community_set:
            theta_tc2p[task_community] = {}

            tem = 0
            for task_pattern in self.task_pattern_set:
                count = 0

                for task in self.c2t[task_community]:
                    for truth in self.label_set:
                        for worker in self.t2wl[task]:
                            worker_community = self.w2c[worker]
                            label = self.t2wl[task][worker]
                            count += self.phi_ig[task][truth] * \
                                     self.phi_dcgkr[task_community][worker_community][truth][label][task_pattern]
                count += self.theta_dir_prior - 1
                theta_tc2p[task_community][task_pattern] = count
                tem += count

            for task_pattern in self.task_pattern_set:
                theta_tc2p[task_community][task_pattern] /= tem

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
                        task_community = self.t2c[task]
                        label = self.w2tl[worker][task]

                        for truth in self.label_set:
                            count += self.phi_ig[task][truth] * \
                                     self.phi_dcgkq[task_community][worker_community][truth][label][worker_pattern]
                count += self.theta_dir_prior - 1
                theta_wc2p[worker_community][worker_pattern] = count
                tem += count

            for worker_pattern in self.worker_pattern_set:
                theta_wc2p[worker_community][worker_pattern] /= tem

        # t2 = time.time()
        # print('The time cost in update theta_wc2p is %e' % (t2 - t1))

        # compute pi_l
        # t1 = time.time()
        for truth in self.label_set:
            pi_l[truth] = {}

            tem = 0
            for label in self.label_set:
                count = 0

                for task, worker in self.l2tw[label]:
                    task_community = self.t2c[task]
                    worker_community = self.w2c[worker]
                    count += self.phi_ig[task][truth] * self.phi_dcgk[task_community][worker_community][truth][label]

                count += self.dir_prior_multiplier * self.probability_truth2label[truth][label]
                pi_l[truth][label] = count
                tem += count

            for label in self.label_set:
                pi_l[truth][label] /= tem

        # t2 = time.time()
        # print('The time cost in update pi_l is %e' % (t2 - t1))

        # compute pi_tp
        # t1 = time.time()
        for task_pattern in self.task_pattern_set:
            pi_tp[task_pattern] = {}

            for truth in self.label_set:
                pi_tp[task_pattern][truth] = {}

                tem = 0
                for label in self.label_set:
                    count = 0

                    for task, worker in self.l2tw[label]:
                        task_community = self.t2c[task]
                        worker_community = self.w2c[worker]
                        count += self.phi_ig[task][truth] * \
                                 self.phi_dcgkr[task_community][worker_community][truth][label][task_pattern]
                    count += self.pi_dir_prior - 1
                    pi_tp[task_pattern][truth][label] = count
                    tem += count

                for label in self.label_set:
                    pi_tp[task_pattern][truth][label] /= tem

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
                        task_community = self.t2c[task]
                        worker_community = self.w2c[worker]

                        count += self.phi_ig[task][truth] * \
                                 self.phi_dcgkq[task_community][worker_community][truth][label][
                                     worker_pattern]
                    count += self.pi_dir_prior - 1
                    pi_wp[worker_pattern][truth][label] = count
                    tem += count

                for label in self.label_set:
                    pi_wp[worker_pattern][truth][label] /= tem

        # t2 = time.time()
        # print('The time cost in update pi_wp is %e' % (t2 - t1))

        self.theta_i2g = theta_i2g
        self.theta_tc2p = theta_tc2p
        self.theta_wc2p = theta_wc2p
        self.pi_tp = pi_tp
        self.pi_wp = pi_wp
        self.pi_l = pi_l

        self.pi_l = self.probability_truth2label

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

                    sum_pi_i = 0
                    for task_pattern in self.task_pattern_set:
                        sum_pi_i += self.theta_tc2p[task_community][task_pattern] * self.pi_tp[task_pattern][truth][
                            label]

                    sum_pi_j = 0
                    worker_community = self.w2c[worker]
                    for worker_pattern in self.worker_pattern_set:
                        sum_pi_j += self.theta_wc2p[worker_community][worker_pattern] * \
                                    self.pi_wp[worker_pattern][truth][label]

                    prod_j *= self.pi_l[truth][label] * sum_pi_i * sum_pi_j

                sum_g += prod_j

            likelihood += math.log(sum_g)

        # prior of theta_g
        for task in self.task_set:
            for truth in self.label_set:
                likelihood += (self.theta_dir_prior - 1) * math.log(self.theta_i2g[task][truth])

        # prior of theta_tc2p
        for task_community in self.task_community_set:
            for task_pattern in self.task_pattern_set:
                likelihood += (self.theta_dir_prior - 1) * math.log(self.theta_tc2p[task_community][task_pattern])

        # prior of theta_wc2p
        for worker_community in self.worker_community_set:
            for worker_pattern in self.worker_pattern_set:
                likelihood += (self.theta_dir_prior - 1) * math.log(self.theta_wc2p[worker_community][worker_pattern])

        # prior of pi_tp
        for task_pattern in self.task_pattern_set:
            for truth in self.label_set:
                for label in self.label_set:
                    likelihood += (self.pi_dir_prior - 1) * math.log(self.pi_tp[task_pattern][truth][label])

        # normalization of pi_wp
        for worker_pattern in self.worker_pattern_set:
            for truth in self.label_set:
                for label in self.label_set:
                    likelihood += (self.pi_dir_prior - 1) * math.log(self.pi_wp[worker_pattern][truth][label])

        # normalization of pi_l
        for truth in self.label_set:
            for label in self.label_set:
                likelihood += (self.pi_dir_prior - 1) * math.log(self.pi_l[truth][label])

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
                    aux += self.phi_ig[task][truth] * math.log(self.pi_l[truth][label])

                    task_community = self.t2c[task]
                    for task_pattern in self.task_pattern_set:
                        aux += self.phi_ig[task][truth] * self.phi_gkdr[task_community][truth][label][
                            task_pattern] * math.log(
                            self.theta_tc2p[task_community][task_pattern] * self.pi_tp[task_pattern][truth][label] /
                            self.phi_gkdr[task_community][truth][label][task_pattern])

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

        for iter in range(self.max_iteration):
            # E-step
            self.update_phi_ig()
            # print('phi_ig is updated')
            self.update_phi_gk()
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
                if task not in t2wl:
                    t2wl[task] = {}
                t2wl[task][worker] = label

        for task in t2wl:
            for worker in t2wl[task]:
                label = t2wl[task][worker]

                gross_labels += 1



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

    def init_info_community(self):

        t2c = {}
        w2c = {}
        c2t = {}
        c2w = {}
        tc2l = {}
        wc2l = {}
        task_community_set = set()
        worker_community_set = set()
        tcwc2l = {}

        # task community
        for task in self.task_set:
            t2c[task] = task

        if hasattr(self, 'task_community_file'):
            with open(self.task_community_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)

                for line in reader:
                    task, community = line
                    if task not in self.task_set:
                        continue
                    t2c[task] = community

        for task in t2c:
            community = t2c[task]
            task_community_set.add(community)

        # worker community
        for worker in self.worker_set:
            w2c[worker] = worker

        if hasattr(self, 'worker_community_file'):
            with open(self.worker_community_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)

                for line in reader:
                    worker, community = line
                    if worker not in self.worker_set:
                        continue
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

        for task in self.t2wl:
            task_community = t2c[task]
            if task_community not in tcwc2l:
                tcwc2l[task_community] = {}

            for worker in self.t2wl[task]:
                worker_community = w2c[worker]
                if worker_community not in tcwc2l[task_community]:
                    tcwc2l[task_community][worker_community] = set()
                label = self.t2wl[task][worker]
                tcwc2l[task_community][worker_community].add(label)

        self.t2c = t2c
        self.w2c = w2c
        self.c2t = c2t
        self.c2w = c2w
        self.tc2l = tc2l
        self.wc2l = wc2l
        self.tcwc2l = tcwc2l
        self.task_community_set = task_community_set
        self.worker_community_set = worker_community_set
        self.num_task_communities = len(task_community_set)
        self.num_worker_communities = len(worker_community_set)

    def get_info_label_relation(self):
        probability_truth2label = {}

        for truth in self.label_set:
            probability_truth2label[truth] = {}
            for label in self.label_set:
                probability_truth2label[truth][label] = self.init_quality if truth == label else (
                                                                                                         1 - self.init_quality) / (
                                                                                                         self.num_labels - 1)

        if hasattr(self, 'label_relation_file'):
            with open(self.label_relation_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)

                for truth in self.label_set:
                    for label in self.label_set:
                        probability_truth2label[truth][label] = 1 if truth == label else 0.05

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
                    # print('task %s in not found in the list of answers' % task)
                    pass
                elif self.t2a[task] == truth:
                    count.append(1)
                else:
                    count.append(0)

        return sum(count) / len(count)


if __name__ == '__main__':
    start_time = time.time()


    # dataset_list = ['CF', 'CF_amt', 'd_Duck_Identification', 'd_jn_product', 'd_sentiment', 'MS', 's4_Dog_data',
    #                 's4_Face_Sentiment_Identification', 's4_Relevance', 's5_AdultContent', 'SP', 'SP_amt',
    #                 'ZenCrowd_all',
    #                 'ZenCrowd_in', 'ZenCrowd_us']
    dataset= 's4_Face_Sentiment_Identification'

    datafile = './datasets/' + dataset + '/answer.csv'
    truth_file = './datasets/' + dataset + '/truth.csv'
    task_community_file = './datasets/' + dataset + '/task_relation.csv'
    worker_community_file = './datasets/' + dataset + '/worker_relation.csv'
    datafile_test = './datasets/' + dataset + '/answer_test.csv'
    datafile_validation = './datasets/' + dataset + '/answer_validation.csv'


    tin = TIN(datafile_test, truth_file=truth_file, beta_1=0.15, beta_2=0.35, beta_3=0.5, worker_community_file = worker_community_file)
    # tin = TIN(datafile, truth_file=truth_file, label_relation_file=label_relation_file)

    print('object has created successfully!')
    tin.run()
    accuracy = tin.get_accuracy()
    print('The accuracy of TIN is %f' % accuracy)

    end_time = time.time()
    print('The time cost is %f' % (end_time - start_time))
