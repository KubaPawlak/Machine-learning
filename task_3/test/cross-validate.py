import logging

from data.movie import train
from task_3.similarity import similarity_function
from task_3.cross_validation import kfold_cross_validation

def _main():
    logging.basicConfig(level=logging.INFO)
    mse_mean, mse_stddev = kfold_cross_validation(train, similarity_function, k=5)
    print(f"MSE = {mse_mean:.2f} (\u03c3={mse_stddev:.2f})")

if __name__ == '__main__':
    _main()