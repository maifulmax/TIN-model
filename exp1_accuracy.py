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
# from multiprocess import Pool
import attr


def get_method_accuracy_in_data(func_name, datafile, truth_file):
    func = globals()[func_name]
    cls = func(datafile, truth_file=truth_file)
    cls.run()
    accuracy = cls.get_accuracy()
    print('The accuracy of %s on data %s  is %f' % (func, datafile, accuracy))

    return accuracy


if __name__ == '__main__':

    # dataset_list = ['CF', 'CF_amt', 'd_Duck_Identification', 'd_jn_product', 'd_sentiment', 'MS', 's4_Dog_data',
    #                 's4_Face_Sentiment_Identification', 's4_Relevance', 's5_AdultContent', 'SP', 'SP_amt', 'ZenCrowd_all',
    #                 'ZenCrowd_in', 'ZenCrowd_us']
    dataset_list = ['CF_amt']
    # method_list = ['BCC', 'CATD', 'CBCC', 'DS', 'GLAD', 'MV', 'PM', 'ZC']
    method_list = ['CATD', 'DS', 'GLAD', 'MV', 'PM', 'ZC']
    # method_list = ['MV']

    f = xlwt.Workbook(encoding='ascii')
    fs = f.add_sheet('accuracy')
    fs.write(0, 0, label='data')

    for i in range(len(dataset_list)):
        fs.write(i + 1, 0, label=dataset_list[i])
    for j in range(len(method_list)):
        fs.write(0, j + 1, label=method_list[j])

    results = []
    pool = Pool()
    for i in range(len(dataset_list)):
        for j in range(len(method_list)):
            dataset = dataset_list[i]
            datafile = './datasets/' + dataset + '/answer_test.csv'
            truth_file = './datasets/' + dataset + '/truth.csv'
            method_name = method_list[j]
            result = pool.apply_async(get_method_accuracy_in_data, args=(method_name, datafile, truth_file))
            results.append(result)

    for k in range(len(results)):
        i = k // len(method_list)
        j = k % len(method_list)
        accuracy = results[k].get()
        fs.write(i + 1, j + 1, label=accuracy)

    f.save('./accuracy_list.xls')
    pool.close()
    pool.join()
