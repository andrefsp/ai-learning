import argparse
import os

import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam
from linear import LinearModel


# Google Cloud ML Engine requires run_experiment.
# ``run_experiment`` is the entry point for ML engine jobs.
# As this is a train job than we instantiate the LinearModel class and we
# train the model from a file passed through hparams and we specify the
# export path to product the model files. (*.pb)


def run_experiment(hparams):
    '''
    Google ML Engine entry point for training job.
    '''
    linear_model = LinearModel(hparams.export_path)
    linear_model.train_from_file(hparams.train_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--train-file',
        help='GCS or local paths to training data',
        default="data/linear.train.csv",
        required=True
    )
    parser.add_argument(
        '--export-path',
        help='GCS or local paths to training data',
        default="./export/",
        required=True
    )
    # passed only in ML engine.
    parser.add_argument(
        '--job-dir',
        help='GCS or local paths to training data',
        default="./job/",
        required=False
    )
    parser.add_argument(
        '--verbosity',
        choices=[
            'DEBUG',
            'ERROR',
            'FATAL',
            'INFO',
            'WARN'
        ],
        default='INFO',
    )

    args = parser.parse_args()

    # Set python level verbosity
    tf.logging.set_verbosity(args.verbosity)
    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf.logging.__dict__[args.verbosity] / 10)

    # Run the training job
    hparams = hparam.HParams(**args.__dict__)
    run_experiment(hparams)
