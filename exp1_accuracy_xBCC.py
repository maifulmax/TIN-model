import csv
import xlwt
from operator import methodcaller
from Methods.BCC.method import BCC
from Methods.CATD.method import CATD
import Methods.CATD.read_distribution as cdis
from Methods.CBCC.method import CBCC
from Methods.DS.method import DS
from Methods.GLAD.method import GLAD
from Methods.MV.method import MV
from Methods.PM.method import PM
from Methods.TIN.method_linear import TIN
from Methods.ZC.method import ZC
from multiprocessing import Pool
import attr

def get_method_accuracy_in_data(func_name, datafile, truth_file):
    func = globals()[func_name]
    cls = func(datafile,truth_file=truth_file)
    cls.run()
    accuracy = cls.get_accuracy()
    print('The accuracy of %s on data %s  is %f' % (func, datafile,  accuracy))

    return accuracy


dataset_list = ['CF', 'CF_amt', 'd_Duck_Identification', 'd_jn_product', 'd_sentiment', 'MS', 's4_Dog_data',
                's4_Face_Sentiment_Identification', 's4_Relevance', 's5_AdultContent', 'SP', 'SP_amt', 'ZenCrowd_all',
                'ZenCrowd_in', 'ZenCrowd_us']
# dataset_list = ['CF', 'CF_amt', 'd_Duck_Identification', 'd_jn_product', 'd_sentiment', 'MS', 's4_Dog_data',
#                 's4_Face_Sentiment_Identification', 's4_Relevance', 's5_AdultContent', 'SP', 'SP_amt', 'ZenCrowd_all',
#                 'ZenCrowd_in', 'ZenCrowd_us']
# method_list = ['BCC', 'CATD', 'CBCC', 'DS', 'GLAD', 'MV', 'PM', 'TIN', 'ZC']
method_list = ['BCC','CBCC']

f = xlwt.Workbook(encoding='ascii')
fs = f.add_sheet('accuracy')
fs.write(0,0,label='data')

for i in range(len(dataset_list)):
    fs.write(i+1, 0, label=dataset_list[i])
for j in range(len(method_list)):
    fs.write(0, j+1, label=method_list[j])

for i in range(len(dataset_list)):
    for j in range(len(method_list)):
        dataset = dataset_list[i]
        datafile = './datasets/' + dataset + '/answer_test.csv'
        truth_file = './datasets/' + dataset + '/truth.csv'
        method_name =  method_list[j]
        result = get_method_accuracy_in_data(method_name, datafile, truth_file)
        fs.write(i + 1, j + 1, label=result)

f.save('./accuracy_BCC_CBCC.xls')
