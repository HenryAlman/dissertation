import pickle
import numpy
import gymnasium
from pathlib import Path
from diss_script import run_evaluation

path_str = "/home/henry/dissertation_logs_actual_2/dissertation_logs/diss_script/"
exp_str = "MAPElites_MEDIUM_MAZE_hex_per_joint/"
folder_str = "2026-08-12_16-31-08_seed-None_90905"


path = Path(path_str+exp_str+folder_str)
num_to_sim = 5
xml_file = "/home/henry/dissertation_5thAug/rangefinder_hex.xml"
env_seed=52
use_saved_emitter_0 = True

run_evaluation(outdir = path, num_to_sim=num_to_sim, xml_file=xml_file, env_seed=env_seed, seed=None, use_saved_emitter_0=use_saved_emitter_0)