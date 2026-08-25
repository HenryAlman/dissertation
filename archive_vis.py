from diss_utils import save_heatmap
import pickle


outdir = "/home/henry/dissertation_logs_actual_2/accurate_heatmaps/"
dirpath = "/home/henry/dissertation_logs_actual_2/dissertation_logs/diss_script/"

paths = {}
paths["ant_uni"] = {}
paths["hex_uni"] = {}
paths["ant_per"] = {}
paths["hex_per"] = {}

paths["ant_uni"]["MAP"] = dirpath+"MAPElites_MEDIUM_MAZE_ant_unilateral/2026-08-12_16-39-09_seed-None_82602/"
paths["ant_per"]["MAP"] = dirpath+"MAPElites_MEDIUM_MAZE_ant_per_joint/2026-08-12_16-39-09_seed-None_88235/"
paths["hex_uni"]["MAP"] = dirpath+"MAPElites_MEDIUM_MAZE_hex_unilateral/2026-08-12_16-39-09_seed-None_97845/"
paths["hex_per"]["MAP"] = dirpath+"MAPElites_MEDIUM_MAZE_hex_per_joint/2026-08-12_16-39-09_seed-None_28509/"
paths["ant_uni"]["BOP"] = "/home/henry/dissertation_logs_actual_3/dissertation_logs/diss_script/BOPElites_MEDIUM_MAZE_ant_unilateral/2026-08-14_11-49-27_seed-None_9743/"
paths["ant_per"]["BOP"] = dirpath+"BOPElites_MEDIUM_MAZE_ant_per_joint/2026-08-12_16-31-37_seed-None_81076/"
paths["hex_uni"]["BOP"] = dirpath+"BOPElites_MEDIUM_MAZE_hex_unilateral/2026-08-12_16-30-52_seed-None_83790/"
paths["hex_per"]["BOP"] = dirpath+"BOPElites_MEDIUM_MAZE_hex_per_joint/2026-08-12_16-30-52_seed-None_56475/"
paths["ant_uni"]["BAY"] = dirpath+"BayesOpt_MEDIUM_MAZE_ant_unilateral/2026-08-12_16-39-09_seed-None_24928/"
paths["ant_per"]["BAY"] = dirpath+"BayesOpt_MEDIUM_MAZE_ant_per_joint/2026-08-12_16-31-37_seed-None_92430/"
paths["hex_uni"]["BAY"] = dirpath+"BayesOpt_MEDIUM_MAZE_hex_unilateral/2026-08-12_16-31-08_seed-None_89056/"
paths["hex_per"]["BAY"] = dirpath+"BayesOpt_MEDIUM_MAZE_hex_per_joint/2026-08-12_16-39-09_seed-None_21333/"


for config,dict in paths.items():
    max = 0
    archives = {}
    for alg,path in dict.items():
        if (alg != "BOP"):
            with open(path+"emitter0.sav", "rb") as f:
                e0 = pickle.load(f)
            archive = e0.archive
            archives[config+"_"+alg] = archive
        else:
            with open(path+"result_archive.sav", "rb") as f:
                archive = pickle.load(f)
            with open(path+"obs_and_predicted_archive.sav", "rb") as f:
                upscale_archive = pickle.load(f)
            archives[config+"_"+alg] = archive
            archives[config+"_"+alg+"_upscale"] = upscale_archive
            if (max < upscale_archive.stats.obj_max):
                max = upscale_archive.stats.obj_max

        if (max < archive.stats.obj_max):
            max = archive.stats.obj_max

    for str,archive in archives.items():
        save_heatmap(archive, outdir + (str+".png"), 0, max)