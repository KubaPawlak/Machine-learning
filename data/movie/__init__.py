from pathlib import Path

import pandas as pd

_csv_dir: Path = Path(__file__).parent / 'raw'

movies: pd.DataFrame = pd.read_csv(_csv_dir / 'movie.csv', sep=';', header=0)
train: pd.DataFrame = pd.read_csv(_csv_dir / 'train.csv', sep=';', header=0)
task: pd.DataFrame = pd.read_csv(_csv_dir / 'task.csv', sep=';', header=0)
