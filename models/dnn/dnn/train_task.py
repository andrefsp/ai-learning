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


def read_file_in_batches(filename, batch_size=100):

    dataframe = pd.read_csv(filename)
    labels = dataframe['label']

    del dataframe['label']

    dataframe = pd.DataFrame(
        preprocessing.normalize(dataframe)
    )

    one_n_labels = np.eye(len(labels.unique()))[labels]

    dataframe = dataframe.join(pd.DataFrame(one_n_labels), rsuffix='_label')

    dataset = dataframe.as_matrix()

    for i in range(len(dataset)):

        batch = dataset[i * batch_size:(i + 1) * batch_size]

        if not len(batch):
            break

        Y = []
        X = []

        for line in batch:
            Y.append(line[-10:])
            X.append(line[:len(line) - 10])

        yield X, Y


def get_eval_data(filename):
    Y = []
    X = []

    for _x, _y in read_file_in_batches(filename, batch_size=1000):
        X.extend(_x)
        Y.extend(_y)

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

            for X, Y in read_file_in_batches(hparams.train_file):
                session.run(
                    model.training,
                    feed_dict={model.x: X, model.y: Y},
                )

            X, Y = get_eval_data(hparams.eval_file)

            print("Epoch(%s) Error: %s " % (epoch, session.run(model.error, feed_dict={model.x: X, model.y: Y})))

        X, Y = get_eval_data(hparams.eval_file)

        for i in range(10):
            print("==================================")
            print("::: %s" % Y[i])
            print("::: %s" % session.run(model.logits, feed_dict={model.x: [X[i], ]}))


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
        default=5,
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
