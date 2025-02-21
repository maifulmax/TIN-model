import xlwt
import csv


dataset_list = ['CF', 'CF_amt', 'd_Duck_Identification', 'd_jn_product', 'd_sentiment', 'MS', 's4_Dog_data',
                's4_Face_Sentiment_Identification', 's4_Relevance', 's5_AdultContent', 'SP', 'SP_amt', 'ZenCrowd_all',
                'ZenCrowd_in', 'ZenCrowd_us', 'MUSIC']

file_handle = xlwt.Workbook(encoding='ascii')
fs = file_handle.add_sheet('stat', cell_overwrite_ok=True)
fs.write(0,0,label='data')

data = {}
for i in range(len(dataset_list)):
    data_name = dataset_list[i]
    datafile = './datasets/' + data_name + '/answer_test.csv'
    with open(datafile, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        task_set = set()
        worker_set = set()
        class_set =set()
        num_label = 0

        for line in reader:
            task, worker, label = line
            task_set.add(task)
            worker_set.add(worker)
            class_set.add(label)
            num_label += 1


    num_task = len(task_set)
    num_worker = len(worker_set)
    num_class = len(class_set)
    fs.write(i+1,0,label=data_name)

    fs.write(0,1,label=r'#task')
    fs.write(i+1,1,label=num_task)

    fs.write(0, 2, label=r'#worker')
    fs.write(i + 1, 2, label=num_worker)

    fs.write(0, 3, label=r'#class')
    fs.write(i + 1, 3, label=num_class)

    fs.write(0, 4, label=r'#label')
    fs.write(i + 1, 4, label=num_label)

    fs.write(0, 5, label=r'#label/#task')
    fs.write(i + 1, 5, label=num_label/num_task)

    fs.write(0, 6, label=r'#label/#worker')
    fs.write(i + 1, 6, label=num_label/num_worker)


file_handle.save('./data_stat_test.xls')