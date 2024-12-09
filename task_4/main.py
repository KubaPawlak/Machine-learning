from data.movie import train
from task_4.model import Model


def _main():
    model = Model(train, n_features=20)
    model.train(learning_rate=0.001)
    print(model._loss())

if __name__ == '__main__':
    _main()