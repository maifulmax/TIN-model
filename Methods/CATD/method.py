import math,csv,random
import numpy as np
import Methods.CATD.read_distribution as cdis
import sys
import os

class CATD:
    def __init__(self,datafile, **kwargs):
        e2wl, w2el, label_set = self.gete2wlandw2el(datafile)
        self.e2wl = e2wl
        self.w2el = w2el
        self.weight = dict()
        self.datatype = 'categorical'
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])

    def examples_truth_calculation(self):
        self.truth = dict()

        if self.datatype == 'continuous':
             for example, worker_label_set in self.e2wl.items():
                temp = 0
                for worker, label in worker_label_set:
                    temp = temp + self.weight[worker] * float(label)

                self.truth[example] = temp

        else:
            for example, worker_label_set in self.e2wl.items():
                temp = dict()
                for worker, label in worker_label_set:
                    if label not in temp:
                        temp[label] = self.weight[worker]
                    else:
                        temp[label] += self.weight[worker]

                max = 0
                for label, num in temp.items():
                    if num > max:
                        max = num

                candidate = []
                for label, num in temp.items():
                    if max == num:
                        candidate.append(label)

                self.truth[example] = random.choice(candidate)

    def workers_weight_calculation(self):

        weight_sum = 0
        for worker, example_label_set in self.w2el.items():
            ns = len(example_label_set)
            if ns <= 30:
                chi_s = self.chi_square_distribution[ns][1-self.alpha/2]
            else:
                chi_s = 0.5 * pow(self.normal_distribution[1-self.alpha/2] + pow(2*ns-1 , 0.5) ,2)
            #print ns, chi_s
            dif = 0
            for example, label in example_label_set:
                if self.datatype == 'continuous':
                    dif = dif + (self.truth[example]-float(label))**2
                else:
                    if self.truth[example]!=label:
                        dif = dif + 1
            # if dif==0:
            #     print(worker, ns, dif, chi_s / (dif + 0.00001))

            self.weight[worker] = chi_s / (dif + 0.000000001)
            weight_sum = weight_sum + self.weight[worker]

        for worker in self.w2el.keys():
            self.weight[worker] = self.weight[worker] / weight_sum


    def Init_truth(self):
        self.truth = dict()

        if self.datatype == 'continuous':
            for example, worker_label_set in self.e2wl.items():
                temp = []
                for _, label in worker_label_set:
                    temp.append(float(label))

                self.truth[example] = np.median(temp)  # using median as intial value
                #self.truth[example] = np.mean(temp)  # using mean as initial value


        else:
            # using majority voting to obtain initial value
            for example, worker_label_set in self.e2wl.items():
                temp = dict()
                for _, label in worker_label_set:
                    if label not in temp:
                        temp[label] = 1
                    else:
                        temp[label] += 1

                max = 0
                for label, num in temp.items():
                    if num > max:
                        max = num

                candidate = []
                for label, num in temp.items():
                    if max == num:
                        candidate.append(label)

                self.truth[example] = random.choice(candidate)


    def run(self,alpha=0.05,iterr=100):
        original_dir = os.getcwd()
        os.chdir(os.path.dirname(__file__))
        self.chi_square_conf, self.chi_square_distribution = cdis.read_chi_square_distribution()
        self.normal_conf,self.normal_distribution = cdis.read_normal_distribution()
        self.alpha = alpha

        self.Init_truth()
        while iterr > 0:
            #print getaccuracy(sys.argv[2], self.truth, datatype)

            self.workers_weight_calculation()
            self.examples_truth_calculation()

            iterr -= 1
        os.chdir(original_dir)

        return self.truth, self.weight



###################################
# The above is the EM method (a class)
# The following are several external functions
###################################

    def get_accuracy(self):
        e2truth = {}
        f = open(self.truth_file, 'r')
        reader = csv.reader(f)
        next(reader)

        for line in reader:
            example, truth = line
            e2truth[example] = truth

        tcount = 0
        count = 0

        for e, ptruth in self.truth.items():

            if e not in e2truth:
                continue

            count += 1

            if self.datatype=='continuous':
                tcount = tcount + (ptruth-float(e2truth[e]))**2 #root of mean squared error
                #tcount = tcount + math.fabs(ptruth-float(e2truth[e])) #mean of absolute error
            else:
                if ptruth == e2truth[e]:
                    tcount += 1

        if self.datatype=='continuous':
            return pow(tcount/count,0.5)  #root of mean squared error
            #return tcount/count  #mean of absolute error
        else:
            return tcount*1.0/count

    def gete2wlandw2el(self, datafile):
        e2wl = {}
        w2el = {}
        label_set=[]

        f = open(datafile, 'r')
        reader = csv.reader(f)
        next(reader)

        for line in reader:
            example, worker, label = line
            if example not in e2wl:
                e2wl[example] = []
            e2wl[example].append([worker,label])

            if worker not in w2el:
                w2el[worker] = []
            w2el[worker].append([example,label])

            if label not in label_set:
                label_set.append(label)

        return e2wl,w2el,label_set


if __name__ == "__main__":

    # if len(sys.argv)>=4 and sys.argv[3] == 'continuous':
    #     datatype = r'continuous'
    # else:
    #     datatype = r'categorical'

    datafile = './datasets/d_Duck_Identification/answer.csv'
    truth_file = './datasets/d_Duck_Identification/truth.csv'

    # datafile = r'./datasets/s4_Dog_data/answer.csv'
    # truth_file=r'./datasets/s4_Dog_data/truth.csv'

    # datafile = r'./datasets/BTSK/data_redun19.csv'
    # truth_file=r'./datasets/BTSK/truth.csv'

    # datafile = sys.argv[1]
    catd = CATD(datafile,truth_file=truth_file)
    catd.run()

    # print(w2q)
    # print(e2lpd)

    # truthfile = sys.argv[2]
    accuracy = catd.get_accuracy()
    print('The accuracy is %f' % accuracy)
