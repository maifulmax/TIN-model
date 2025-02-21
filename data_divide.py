import csv
import random
# dataset_list = ['CF', 'CF_amt', 'd_Duck_Identification', 'd_jn_product', 'd_sentiment', 'MS', 's4_Dog_data',
#                 's4_Face_Sentiment_Identification', 's4_Relevance', 's5_AdultContent', 'SP', 'SP_amt', 'ZenCrowd_all',
#                 'ZenCrowd_in', 'ZenCrowd_us', 'MUSIC']
dataset_list = ['CF_amt']



for i in range(len(dataset_list)):
    dataset = dataset_list[i]
    datafile = './datasets/' + dataset + '/answer.csv'
    truth_file = './datasets/' + dataset + '/truth.csv'
    label_relation_file = './datasets/' + dataset + '/label_relation.csv'
    datafile_test = './datasets/' + dataset + '/answer_test.csv'
    datafile_validation = './datasets/' + dataset + '/answer_validation.csv'
    with open(datafile, 'r') as file:
        reader = csv.reader(file)
        head = next(reader)
        file_test = open(datafile_test, 'w', newline='')
        writer_test = csv.writer(file_test)
        writer_test.writerow(head)
        file_validation = open(datafile_validation, 'w', newline='')
        writer_validation = csv.writer(file_validation)
        writer_validation.writerow(head)
        for line in reader:
            if random.random() < 0.5:
                writer_validation.writerow(line)
            else:
                writer_test.writerow(line)
        file_test.close()
        file_validation.close()