import csv


def get_realation(datafile, truth_file, task_relation_dir, worker_relation_dir, label_relation_dir):
    with open(truth_file, 'r') as f1:
        truth_list = {}
        reader = csv.reader(f1)
        next(reader)
        for task, truth in reader:
            truth_list[task] = truth

    with open(datafile, 'r') as f2:
        reader = csv.reader(f2)
        next(reader)
        task_score = {}
        task_num_vote = {}
        worker_score = {}
        worker_num_vote = {}
        label_matrix = {}
        label_relation = {}

        for task, worker, label in reader:
            if task not in truth_list:
                continue
            truth = truth_list[task]
            if truth not in label_matrix:
                label_matrix[truth] = {}
                label_matrix[truth][label] = 1
            if label not in label_matrix[truth]:
                label_matrix[truth][label] = 1

            label_matrix[truth][label] += 1

            if task not in task_score:
                task_score[task] = 0
                task_num_vote[task] = 0
            if worker not in worker_score:
                worker_score[worker] = 0
                worker_num_vote[worker] = 0

            if label == truth_list[task]:
                task_score[task] = (task_score[task] * task_num_vote[task] + 1) / (task_num_vote[task] + 1)
                task_num_vote[task] += 1
                worker_score[worker] = (worker_score[worker] * worker_num_vote[worker] + 1) / (
                            worker_num_vote[worker] + 1)
                worker_num_vote[worker] += 1
            else:
                task_score[task] = (task_score[task] * task_num_vote[task]) / (task_num_vote[task] + 1)
                task_num_vote[task] += 1
                worker_score[worker] = (worker_score[worker] * worker_num_vote[worker]) / (worker_num_vote[worker] + 1)
                worker_num_vote[worker] += 1

        for truth in label_matrix:
            label_relation[truth] = {}
            sum = 0
            for label in label_matrix[truth]:
                sum += label_matrix[truth][label]

            for label in label_matrix[truth]:
                label_relation[truth][label] = label_matrix[truth][label] / sum

        task_relation = {}
        worker_relation = {}
        NUM_COMMUNITY = 20
        for task in task_score:
            task_relation[task] = 'community' + str(int(task_score[task] * NUM_COMMUNITY))
        for worker in worker_score:
            worker_relation[worker] = 'community' + str(int(worker_score[worker] * NUM_COMMUNITY))

        task_num_vote_list = sorted(task_num_vote.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        worker_num_vote_list = sorted(worker_num_vote.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

        with open(task_relation_dir, 'w', newline='', encoding='UTF-8') as fw1:
            writer = csv.writer(fw1)
            writer.writerow(['task', 'community'])
            NUM_TASK_RELATION = len(task_num_vote_list) * 0.1
            k = 0
            for (task, num_vote) in task_num_vote_list:
                # if k > NUM_TASK_RELATION:
                #     break
                # writer.writerow([task, task_relation[task]])
                # k += 1
                writer.writerow([task, truth_list[task]])

        with open(worker_relation_dir, 'w', newline='', encoding='UTF-8') as fw2:
            writer = csv.writer(fw2)
            writer.writerow(['worker', 'community'])
            NUM_WORKER_RELATION = len(worker_num_vote_list) * 0.1
            k = 0
            for (worker, num_vote) in worker_num_vote_list:
                if k > NUM_WORKER_RELATION:
                    break
                writer.writerow([worker, worker_relation[worker]])
                k += 1

        with open(label_relation_dir, 'w', newline='', encoding='UTF-8') as fw3:
            writer = csv.writer(fw3)
            writer.writerow(['truth', 'label', 'probability'])
            for truth in label_relation:
                for label in label_relation[truth]:
                    writer.writerow([truth, label, label_relation[truth][label]])


if __name__ == '__main__':

    dataset_list = ['CF', 'CF_amt', 'd_Duck_Identification', 'd_jn_product', 'd_sentiment', 'MS', 's4_Dog_data',
                    's4_Face_Sentiment_Identification', 's4_Relevance', 's5_AdultContent', 'SP', 'SP_amt',
                    'ZenCrowd_all',
                    'ZenCrowd_in', 'ZenCrowd_us', 'MUSIC']
    # dataset_list = ['CF']

    for i in range(len(dataset_list)):
        dataset = dataset_list[i]
        datafile = './datasets/' + dataset + '/answer_validation.csv'
        truth_file = './datasets/' + dataset + '/truth.csv'
        task_relation_dir = './datasets/' + dataset + '/task_relation.csv'
        worker_relation_dir = './datasets/' + dataset + '/worker_relation.csv'
        label_relation_dir = './datasets/' + dataset + '/label_relation.csv'

        get_realation(datafile, truth_file, task_relation_dir, worker_relation_dir, label_relation_dir)
