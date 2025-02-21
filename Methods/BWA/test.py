import numpy as np
import pandas as pd
import scipy.sparse as ssp


def bwa_binary(y_exists_ij, y_is_one_ij, W_i, lambda_, a_v, adj_coef):
    N_j = y_exists_ij.sum(axis=0)
    z_i = y_is_one_ij.sum(axis=-1) / y_exists_ij.sum(axis=-1)

    b_v = a_v * W_i.dot(np.multiply(z_i, 1 - z_i)) / y_exists_ij.sum() * adj_coef
    for _ in range(500):
        last_z_i = z_i.copy()

        mu = z_i.mean()
        v_j = (a_v + N_j) / (b_v + (y_exists_ij.multiply(z_i) - y_is_one_ij).power(2).sum(0))
        z_i = (lambda_ * mu + y_is_one_ij.dot(v_j.T)) / (lambda_ + y_exists_ij.dot(v_j.T))

        if np.allclose(last_z_i, z_i, rtol=1e-3): break
    return z_i.A1


def bwa(tuples, a_v=15, lambda_=1, prior_correction=True):
    num_items, num_workers, num_classes = tuples.max(axis=0) + 1
    num_labels = tuples.shape[0]
    W_i = np.bincount(tuples[:, 0])

    adj_coef = 4 * (1 - 1 / num_classes) if prior_correction else 1

    y_exists_ij = ssp.coo_matrix((np.ones(num_labels), tuples[:, :2].T),
                                 shape=(num_items, num_workers), dtype=np.bool).tocsr()
    y_is_one_kij = []
    for k in range(num_classes):
        selected = (tuples[:, 2] == k)
        y_is_one_kij.append(ssp.coo_matrix((np.ones(selected.sum()), tuples[selected, :2].T),
                                           shape=(num_items, num_workers), dtype=np.bool).tocsr())
    z_ik = np.empty((num_items, num_classes))
    for k in range(num_classes):
        z_ik[:, k] = bwa_binary(y_exists_ij, y_is_one_kij[k], W_i, lambda_, a_v, adj_coef)
    return z_ik


def get_acc(predictions, df_truth):
    score = (predictions == predictions.max(axis=1, keepdims=True)).astype(np.float)
    score /= score.sum(axis=1, keepdims=True)
    return score[df_truth.item.values, df_truth.truth.values].sum() / df_truth.shape[0]


if __name__ == '__main__':
    records = []
    # dataset_list = ['CF', 'CF_amt', 'd_Duck_Identification', 'd_jn_product', 'd_sentiment', 'MS', 's4_Dog_data',
    #                 's4_Face_Sentiment_Identification', 's4_Relevance', 's5_AdultContent', 'SP', 'SP_amt',
    #                 'ZenCrowd_all',
    #                 'ZenCrowd_in', 'ZenCrowd_us', 'MUSIC']
    dataset_list = ['d_Duck_Identification', 'd_jn_product', 'd_sentiment']

    for dataset in dataset_list:
        datafile = './datasets/' + dataset + '/answer_test.csv'
        truth_file = './datasets/' + dataset + '/truth.csv'
        df_label = pd.read_csv(datafile)
        df_label = df_label.drop_duplicates(keep='first')
        prediction_ik = bwa(df_label.values)

        df_truth = pd.read_csv(truth_file)
        records.append((dataset, get_acc(prediction_ik, df_truth)))
        print('%-10s %g' % records[-1])
