import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam
from dnn import DNN
from sklearn import preprocessing

# Google Cloud ML Engine requires run_experiment.
# ``run_experiment`` is the entry point for ML engine jobs.
# As this is a train job than we instantiate the LinearModel class and we
# train the model from a file passed through hparams and we specify the
# export path to product the model files. (*.pb)


def get_train_data(filename):

    dataframe = pd.read_csv(filename)
    labels = dataframe['label']

    del dataframe['label']

    dataframe = pd.DataFrame(
        preprocessing.normalize(dataframe)
    )

    one_n_labels = np.eye(len(labels.unique()))[labels]

    dataframe = dataframe.join(pd.DataFrame(one_n_labels), rsuffix='_label')

    for line in dataframe.as_matrix():
        Y = [line[-10:], ]
        X = [line[:len(line) - 10], ]
        yield X, Y


def get_eval_data(filename):
    dataframe = pd.read_csv(filename)
    labels = dataframe['label']

    del dataframe['label']

    dataframe = pd.DataFrame(
        preprocessing.normalize(dataframe)
    )

    one_n_labels = np.eye(len(labels.unique()))[labels]

    dataframe = dataframe.join(pd.DataFrame(one_n_labels), rsuffix='_label')

    X = []
    Y = []
    for line in dataframe.as_matrix():
        Y.append(line[-10:])
        X.append(line[:len(line) - 10])

    return X, Y


def run_experiment(hparams):
    '''
    Google ML Engine entry point for training job.
    '''

    model = DNN(784, 10)

    init = tf.global_variables_initializer()

    with tf.Session() as session:

        session.run(init)

        for epoch in range(0, hparams.epochs):

            for X, Y in get_train_data(hparams.train_file):
                session.run(
                    model.training,
                    feed_dict={model.x: X, model.y: Y},
                )

            X, Y = get_eval_data(hparams.eval_file)

            print("Epoch(%s) Error: %s " % (epoch, session.run(model.error, feed_dict={model.x: X, model.y: Y})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--train-file',
        help='GCS or local paths to training data',
        default="data/train.csv",
        required=False
    )

    parser.add_argument(
        '--eval-file',
        help='GCS or local paths to training data',
        default="data/eval.csv",
        required=False
    )

    parser.add_argument(
        '--export-path',
        help='GCS or local paths to training data',
        default="./export/",
        required=False,
    )

    parser.add_argument(
        '--epochs',
        help='GCS or local paths to training data',
        default=100,
        required=False,
    )

    # passed only in ML engine.
    parser.add_argument(
        '--job-dir',
        help='GCS or local paths to training data',
        default="./job/",
        required=False
    )

    args = parser.parse_args()

    # Run the training job
    hparams = hparam.HParams(**args.__dict__)
    run_experiment(hparams)
