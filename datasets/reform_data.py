import csv
import os

file = 'ZenCrowd_us'
data_file = file + '.csv'
answer_file = file + '/answer.csv'
truth_file = file + '/truth.csv'

if not os.path.exists(file):
    os.makedirs(file)

answer_set = set()
truth_set = set()

with open(data_file,'r', encoding='UTF-8') as f:
    reader = csv.reader(f)
    for line in reader:
        worker, task, answer, truth = line[:4]
        answer_set.add((task, worker, answer))
        truth_set.add((task, truth))

with open(answer_file,'w', newline='',encoding='UTF-8') as f1:
    writer = csv.writer(f1)
    writer.writerow(['task', 'worker', 'answer'])
    for i in answer_set:
        writer.writerow(i)

with open(truth_file,'w', newline='', encoding='UTF-8') as f2:
    writer = csv.writer(f2)
    writer.writerow(['task', 'truth'])
    for i in truth_set:
        writer.writerow(i)