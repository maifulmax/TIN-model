import random
import os
import subprocess
import csv

class BCC(object):

    def __init__(self, datafile, **kwargs):
        self.datafile =datafile
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])

    def run(self):
        sep = ","
        exec_cs = True
        answer_list = []
        with open(self.datafile) as f:
            f.readline()
            for line in f:
                if not line:
                    continue
                parts = line.strip().split(sep)
                item_name, worker_name, worker_label = parts[:3]
                answer_list.append([worker_name, item_name, worker_label])

        original_dir = os.getcwd()
        os.chdir(os.path.dirname(__file__))

        with open("Data/CF.csv", "w") as f:
            for piece in answer_list:
                f.write(",".join(piece) + "\n")

        if exec_cs:
            subprocess.getoutput("/bin/rm Results/endpoints.csv")
            os.system('export PATH=/home/LAB/hantao/mono/bin:$PATH;mono CommunityBCCSourceCode.exe')
        e2lpd = {}
        with open("Results/endpoints.csv", 'r') as f:
            for line in f:
                parts = line.strip().split(sep)
                e2lpd[parts[0]] = {}
                for i, v in enumerate(parts[1:]):
                    e2lpd[parts[0]][str(i)] = float(v)

        self.e2lpd = e2lpd
        os.chdir(original_dir)

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

if __name__ == '__main__':

    sep = ","
    exec_cs = True

    # if len(sys.argv) != 3:
    #     print "usage: %s %s %s" % (sys.argv[0], "answer_file", "truth_file")
    datafile = './datasets/MUSIC/answer.csv'
    truth_file = './datasets/MUSIC/truth.csv'

    # datafile = r'./datasets/s4_Dog_data/answer.csv'
    # truth_file=r'./datasets/s4_Dog_data/truth.csv'

    # datafile = r'./datasets/BTSK/data_redun19.csv'
    # truth_file=r'./datasets/BTSK/truth.csv'

    # datafile = 'answer.csv'
    # answer_filename = sys.argv[1]

    bcc = BCC(datafile, truth_file=truth_file)
    bcc.run()
    accuracy =bcc.get_accuracy()

    print('The accuracy is %f' % accuracy)