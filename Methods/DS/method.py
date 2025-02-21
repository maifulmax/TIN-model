import math
import csv
import random
import sys

class DS:
    def __init__(self, datafile, **kwargs):
        e2wl, w2el, label_set = self.gete2wlandw2el(datafile)
        self.e2wl = e2wl
        self.w2el = w2el
        self.workers = self.w2el.keys()
        self.label_set = label_set
        self.initalquality = 0.7
        for kw in kwargs:
            setattr(self,kw,kwargs[kw])
             


# E-step
    def Update_e2lpd(self):
        self.e2lpd = {}

        for example, worker_label_set in self.e2wl.items():
            lpd = {}
            total_weight = 0

            for tlabel, prob in self.l2pd.items():
                weight = prob
                for (w, label) in worker_label_set:
                    weight *= self.w2cm[w][tlabel][label]
                
                lpd[tlabel] = weight
                total_weight += weight
                

            for tlabel in lpd:
                if total_weight == 0:
                    # uniform distribution 
                    lpd[tlabel] = 1.0/len(self.label_set)
                else:
                    lpd[tlabel] = lpd[tlabel]*1.0/total_weight
            
            self.e2lpd[example] = lpd



#M-step

    def Update_l2pd(self):
        for label in self.l2pd:
            self.l2pd[label] = 0

        for _, lpd in self.e2lpd.items():
            for label in lpd:
                self.l2pd[label] += lpd[label]

        for label in self.l2pd:
            self.l2pd[label] *= 1.0/len(self.e2lpd)


            
    def Update_w2cm(self):

        for w in self.workers:
            for tlabel in self.label_set:
                for label in self.label_set:
                    self.w2cm[w][tlabel][label] = 0


        w2lweights = {}
        for w in self.w2el:
            w2lweights[w] = {}
            for label in self.label_set:
                w2lweights[w][label] = 0
            for example, _ in self.w2el[w]:
                for label in self.label_set:
                    w2lweights[w][label] += self.e2lpd[example][label]

            
            for tlabel in self.label_set:

                if w2lweights[w][tlabel] == 0:
                    for label in self.label_set:
                        if tlabel == label:
                            self.w2cm[w][tlabel][label] = self.initalquality
                        else:
                            self.w2cm[w][tlabel][label] = (1-self.initalquality)*1.0/(len(self.label_set)-1)

                    continue

                for example, label in self.w2el[w]:
                    
                        self.w2cm[w][tlabel][label] += self.e2lpd[example][tlabel]*1.0/w2lweights[w][tlabel]



        return self.w2cm
                    

    
    

                     

#initialization
    def Init_l2pd(self):
        #uniform probability distribution
        l2pd = {}
        for label in self.label_set:
            l2pd[label] = 1.0/len(self.label_set)
        return l2pd

    def Init_w2cm(self):
        w2cm = {}
        for worker in self.workers:
            w2cm[worker] = {}
            for tlabel in self.label_set:
                w2cm[worker][tlabel] = {}
                for label in self.label_set:
                    if tlabel == label:
                        w2cm[worker][tlabel][label] = self.initalquality
                    else:
                        w2cm[worker][tlabel][label] = (1-self.initalquality)/(len(self.label_set)-1)

        return w2cm

    def run(self, iter = 50):
        
        self.l2pd = self.Init_l2pd()
        self.w2cm = self.Init_w2cm()

        while iter > 0:
            # E-step
            self.Update_e2lpd() 

            # M-step
            self.Update_l2pd()
            self.Update_w2cm()

            # compute the likelihood
            # print self.computelikelihood()

            iter -= 1
        
        return self.e2lpd, self.w2cm


    def computelikelihood(self):
        
        lh = 0

        for _, worker_label_set in self.e2wl.items():
            temp = 0
            for tlabel, prior in self.l2pd.items():
                inner = prior
                for worker, label in worker_label_set:
                    inner *= self.w2cm[worker][tlabel][label]
                temp += inner
            
            lh += math.log(temp)
        
        return lh


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
        e2lpd = self.e2lpd
        for e in e2lpd:

            if e not in e2truth:
                continue

            temp = 0
            for label in e2lpd[e]:
                if temp < e2lpd[e][label]:
                    temp = e2lpd[e][label]

            candidate = []

            for label in e2lpd[e]:
                if temp == e2lpd[e][label]:
                    candidate.append(label)

            truth = random.choice(candidate)

            count += 1

            if truth == e2truth[e]:
                tcount += 1

        return tcount * 1.0 / count


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
    # datafile = './datasets/d_Duck_Identification/answer.csv'
    # truth_file = './datasets/d_Duck_Identification/truth.csv'

    # datafile = r'./datasets/s4_Dog_data/answer.csv'
    # truth_file=r'./datasets/s4_Dog_data/truth.csv'

    # datafile = r'./datasets/BTSK/data_redun19.csv'
    # truth_file=r'./datasets/BTSK/truth.csv'

    # datafile = r'D:\python_workspace\TIN\datasets\d_jn_product\answer.csv'
    # truth_file = r'D:\python_workspace\TIN\datasets\d_jn_product\truth.csv'

    # datafile = r'D:\python_workspace\TIN\datasets\f201_Emotion_FULL\answer.csv'
    # truth_file = r'D:\python_workspace\TIN\datasets\f201_Emotion_FULL\truth.csv'

    # datafile = 'D:/python_workspace/TIN/datasets/MS/answer.csv'
    # truth_file = 'D:/python_workspace/TIN/datasets/MS/truth.csv'

    datafile = r'./datasets/SP/answer.csv'
    truth_file = r'./datasets/SP/truth.csv'

    em = DS(datafile, truth_file=truth_file)
    em.run()
    # print(w2cm)
    # print(e2lpd)

    # truthfile = r'./truth.csv'
    accuracy = em.get_accuracy()
    print(accuracy)
    

