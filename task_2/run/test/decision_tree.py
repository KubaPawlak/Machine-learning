import logging

from task_2.run.decision_tree import DecisionTreeSubmissionGenerator
from task_2.util import Validator

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sg = DecisionTreeSubmissionGenerator(None)
    validator = Validator(sg, num_runs=1)
    validator.run()