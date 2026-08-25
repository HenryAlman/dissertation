from pathlib import Path
import pandas
import pickle
import ribs
import numpy as np

from diss_utils import save_metrics

folder_path = "/home/henry/dissertation_logs_actual_3/dissertation_logs"

for path in Path(folder_path).rglob("obs_and_predicted_archive.sav"):
    if path.is_file():

        with open(path, "rb") as f:
            obs_and_predicted_archive = pickle.load(f)

        result_archive_path = path.parent / "result_archive.sav"

        with open(result_archive_path, "rb") as f:
            result_archive = pickle.load(f)

        metrics = {
                "Max Score": {
                    "result": [result_archive.stats.obj_max],
                    "plus_predicted": [obs_and_predicted_archive.stats.obj_max],
                },
                "Mean Score": {
                    "result": [result_archive.stats.obj_mean],
                    "plus_predicted": [obs_and_predicted_archive.stats.obj_mean],
                },
                "Archive Size": {
                    "result": [len(result_archive)],
                    "plus_predicted": [len(obs_and_predicted_archive)],
                },
                "QD Score": {
                    "result": [result_archive.stats.qd_score],
                    "plus_predicted": [obs_and_predicted_archive.stats.qd_score],
                },
                "Coverage": {
                    "result": [result_archive.stats.coverage],
                    "plus_predicted": [obs_and_predicted_archive.stats.coverage],
                },
            }

        df = pandas.DataFrame(metrics).T
        df['result'] = df['result'].apply(lambda x: np.item(x) if hasattr(x, 'item') else x[0])
        df['plus_predicted'] = df['plus_predicted'].apply(lambda x: np.item(x) if hasattr(x, 'item') else x[0])

        df.index.name = 'Metric'

        save_path = path.parent / "upscale_comparison.csv"

        df.to_csv(save_path, index=True)