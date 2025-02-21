import csv
import random


class MV:

    def __init__(self, datafile, **kwargs):
        # change settings
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])

        # initialise datafile
        self.datafile = datafile
        t2wl, task_set, label_set = self.get_info_data()
        self.t2wl = t2wl
        self.task_set = task_set
        self.num_tasks = len(task_set)
        self.label_set = label_set
        self.num_labels = len(label_set)

    def get_info_data(self):
        if not hasattr(self, 'datafile'):
            raise BaseException('There is no datafile!')

        t2wl = {}
        task_set = set()
        label_set = set()

        f = open(self.datafile, 'r')
        reader = csv.reader(f)
        next(reader)

        for line in reader:
            task, worker, label = line

            if task not in t2wl:
                t2wl[task] = {}
            t2wl[task][worker] = label

            if task not in task_set:
                task_set.add(task)

            if label not in label_set:
                label_set.add(label)


        return t2wl,task_set,label_set

    def get_accuracy(self):
        if not hasattr(self, 'truth_file'):
            raise BaseException('There is no truth file!')
        if not hasattr(self, 't2a'):
            raise BaseException('There is no aggregated answers!')

        count = []
        f = open(self.truth_file,'r')
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

        return sum(count)/len(count)

    def run(self):
        # initialization
        count = {}
        for task in self.task_set:
            count[task] = {}
            for label in self.label_set:
                count[task][label] = 0

        # compute
        for task in self.task_set:
            for worker in self.t2wl[task]:
                label = self.t2wl[task][worker]
                count[task][label] += 1
        t2a = {}
        for task in self.task_set:
            t2a[task] = random.choice(list(self.label_set))
            for label in self.label_set:
                if count[task][label] > count[task][t2a[task]]:
                    t2a[task] = label
        self.t2a = t2a

        # return self.expand(e2lpd)
        return t2a




if __name__ == "__main__":
    datafile = r'./datasets/d_Duck_Identification/answer.csv'
    truth_file=r'./datasets/d_Duck_Identification/truth.csv'

    # datafile = r'D:\python_workspace\TIN\datasets\d_jn-product\answer.csv'
    # truth_file = r'D:\python_workspace\TIN\datasets\d_jn-product\truth.csv'

    # datafile = r'D:\python_workspace\TIN\datasets\s4_Dog_data\answer.csv'
    # truth_file=r'D:\python_workspace\TIN\datasets\s4_Dog_data\truth.csv'

    # datafile = r'D:\python_workspace\TIN\datasets\f201_Emotion_FULL\answer.csv'
    # truth_file = r'D:\python_workspace\TIN\datasets\f201_Emotion_FULL\truth.csv'

    # datafile = r'D:\python_workspace\TIN\datasets\MS\answer.csv'
    # truth_file = r'D:\python_workspace\TIN\datasets\MS\truth.csv'

    mv = MV(datafile, truth_file=truth_file)

    print('object has created successfully!')
    mv.run()
    accuracy = mv.get_accuracy()
    print('The accuracy of mv is %f' % accuracy)
    

