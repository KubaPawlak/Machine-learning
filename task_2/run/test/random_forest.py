import logging

from task_2.run.random_forest import RandomForestSubmissionGenerator
from task_2.util import Validator

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sg = RandomForestSubmissionGenerator(None)
    validator = Validator(sg, num_runs=1)
    validator.run()