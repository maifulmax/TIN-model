from Methods.TIN.method_linear import TIN
from multiprocessing import Pool
import numpy as np
import csv
import os

def test_TIN_hyperparameters(datafile, truth_file, **kwargs):
    # print('Process %s is started' % os.getpid())

    kwargs.update(truth_file=truth_file)


    tin = TIN(datafile, **kwargs)
    # print('object has created successfully!')
    tin.run()
    accuracy = tin.get_accuracy()
    print('The accuracy is %f with theta_dir_prior %f, pi_dir_prior %f, dir_prior_multiplier %f' % (accuracy, tin.theta_dir_prior, tin.pi_dir_prior, tin.dir_prior_multiplier))
    return [tin.theta_dir_prior, tin.pi_dir_prior, tin.dir_prior_multiplier, accuracy]


if __name__ == '__main__':

    # datafile = r'./datasets/test/answer.csv'
    # truth_file = r'./datasets/test/truth.csv'

    # datafile = r'./datasets/d_Duck_Identification/answer.csv'
    # truth_file = r'./datasets/d_Duck_Identification/truth.csv'

    # datafile = r'./datasets/s4_Dog_data/answer.csv'
    # truth_file=r'./datasets/s4_Dog_data/truth.csv'

    # datafile = r'./datasets/BTSK/data_redun19.csv'
    # truth_file = r'./datasets/BTSK/truth.csv'
    # label_relation_file = r'./datasets/BTSK/label_relation.csv'

    # datafile = r'./datasets/d_jn_product/answer.csv'
    # truth_file = r'./datasets/d_jn_product/truth.csv'

    # datafile = r'D:\python_workspace/TIN/datasets/f201_Emotion_FULL/answer.csv'
    # truth_file = r'D:\python_workspace/TIN/datasets/f201_Emotion_FULL/truth.csv'

    # datafile = r'./datasets/MS/answer.csv'
    # truth_file = r'./datasets/MS/truth.csv'
    dataset = 'd_jn_product'
    datafile = './datasets/' + dataset +'/answer.csv'
    truth_file = './datasets/' + dataset + '/truth.csv'

    theta_dir_prior_list = [1 + 1e-4, 2, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10]
    pi_dir_prior_list = [1 + 1e-4, 2, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10]
    dir_prior_multiplier_list = [0, 1, 10, 50, 100, 500, 1000]

    pool = Pool()
    results = []
    for theta_dir_prior in theta_dir_prior_list:
        for pi_dir_prior in pi_dir_prior_list:
            for dir_prior_multiplier in dir_prior_multiplier_list:
                accuracy = pool.apply_async(test_TIN_hyperparameters, args=(datafile, truth_file), kwds={'theta_dir_prior':theta_dir_prior, 'pi_dir_prior':pi_dir_prior, 'dir_prior_multiplier':dir_prior_multiplier})
                results.append(accuracy)

    head = ['theta_dir_prior', 'pi_dir_prior', 'dir_prior_multiplier', 'accuracy']
    result_file_name = 'accuracy_prior_'+ dataset + '.csv'
    f = open(result_file_name,'w', newline='')
    writer = csv.writer(f)
    writer.writerow(head)
    for result in results:
        writer.writerow(result.get())
    f.close()
    pool.close()
    pool.join()
    # run(max_iteration=7)