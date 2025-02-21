from Methods.TIN.method_linear import TIN
from multiprocessing import Pool
import numpy as np
import csv
import time

def test_TIN_hyperparameters(datafile, truth_file, **kwargs):
    # print('Process %s is started' % os.getpid())

    kwargs.update(truth_file=truth_file)


    tin = TIN(datafile, **kwargs)
    # print('object has created successfully!')
    tin.run()
    accuracy = tin.get_accuracy()
    print('The accuracy is %f with beta_1 %f, beta_2 %f, beta_3 %f, init_quality %f' % (accuracy, tin.beta_1, tin.beta_2, tin.beta_3, tin.init_quality))

    return [tin.beta_1, tin.beta_2, tin.beta_3, tin.init_quality, accuracy]


if __name__ == '__main__':
    start_time = time.time()
    # datafile = r'./datasets/test/answer.csv'
    # truth_file = r'./datasets/test/truth.csv'

    # datafile = r'./datasets/d_Duck_Identification/answer.csv'
    # truth_file = r'./datasets/d_Duck_Identification/truth.csv'

    # datafile = r'./datasets/s4_Dog_data/answer.csv'
    # truth_file=r'./datasets/s4_Dog_data/truth.csv'

    # datafile = r'./datasets/BTSK/data_redun10.csv'
    # truth_file = r'./datasets/BTSK/truth.csv'
    # label_relation_file = r'./datasets/BTSK/label_relation.csv'

    # datafile = r'./datasets/d_jn_product/answer.csv'
    # truth_file = r'./datasets/d_jn_product/truth.csv'

    # datafile = r'D:\python_workspace/TIN/datasets/f201_Emotion_FULL/answer.csv'
    # truth_file = r'D:\python_workspace/TIN/datasets/f201_Emotion_FULL/truth.csv'



    # datafile = r'./datasets/MS/answer.csv'
    # truth_file = r'./datasets/MS/truth.csv'

    dataset = 'd_jn_product'
    datafile = './datasets/' + dataset + '/answer.csv'
    truth_file = './datasets/' + dataset + '/truth.csv'

    theta_dir_prior_list = [1 + 1e-4, 2, 2 ** 2, 2 ** 3, 2 ** 4, 2 ** 5, 2 ** 6, 2 ** 7, 2 ** 8, 2 ** 9, 2 ** 10]
    pi_dir_prior_list = [1 + 1e-4, 2, 2 ** 2, 2 ** 3, 2 ** 4, 2 ** 5, 2 ** 6, 2 ** 7, 2 ** 8, 2 ** 9, 2 ** 10]
    dir_prior_multiplier_list = [0, 1, 10, 50, 100, 500, 1000]


    pool = Pool()
    results = []
    for beta_1 in np.linspace(0,1,21):
        for beta_2 in np.linspace(0, 1-beta_1, int(round((1.05-beta_1)/0.05))):
            for init_quality in np.linspace(0.5,0.95,10):
                beta_3 = 1 - beta_1 -beta_2
                accuracy = pool.apply_async(test_TIN_hyperparameters, args=(datafile, truth_file), kwds={'beta_1':beta_1, 'beta_2':beta_2, 'beta_3':beta_3, 'init_quality':init_quality})
                results.append(accuracy)

    head = ['beta_1', 'beta_2', 'beta_3', 'init_quality', 'accuracy']
    result_file_name = 'accuracy_beta_'+ dataset + '.csv'
    f = open(result_file_name,'w', newline='')
    writer = csv.writer(f)
    writer.writerow(head)
    for result in results:
        writer.writerow(result.get())
    f.close()
    pool.close()
    pool.join()

    end_time = time.time()

    print('The time cost is %f' % (end_time - start_time))