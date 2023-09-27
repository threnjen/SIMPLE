import os
import sys
import random
import csv
import time
import numpy as np

from mpi4py import MPI

from shutil import rmtree
from stable_baselines.ppo1 import PPO1
from stable_baselines.common.policies import MlpPolicy

from utils.register import get_network_arch

import config

from stable_baselines import logger


def write_results(players, game, games, episode_length):
    out = {
        "game": game,
        "games": games,
        "episode_length": episode_length,
        "p1": players[0].name,
        "p2": players[1].name,
        "p1_points": players[0].points,
        "p2_points": np.sum([x.points for x in players[1:]]),
    }

    if not os.path.exists('viz/results.csv'):
        with open('viz/results.csv', "a") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=out.keys())
            writer.writeheader()

    with open('viz/results.csv', "a") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=out.keys())
        writer.writerow(out)


def load_model(env, name):
    filename = f"zoo/{env.name}/{name}"
    if os.path.exists(filename):
        logger.info(f"Loading {name}")
        cont = True
        while cont:
            try:
                ppo_model = PPO1.load(filename, env=env)
                cont = False
            except Exception as e:
                time.sleep(5)
                print(e)

    elif name == "base.zip":
        cont = True
        while cont:
            try:
                rank = MPI.COMM_WORLD.Get_rank()
                if rank == 0:
                    ppo_model = PPO1(get_network_arch(env.name), env=env)
                    logger.info(f"Saving base.zip PPO model...")
                    ppo_model.save(f"zoo/{env.name}/base.zip")
                else:
                    ppo_model = PPO1.load(f"zoo/{env.name}/base.zip", env=env)

                cont = False
            except IOError as e:
                sys.exit(f"Check zoo/{env.name}/ exists and read/write permission granted to user")
            except Exception as e:
                logger.error(e)
                time.sleep(2)

    else:
        raise Exception(f"\n{filename} not found")

    return ppo_model


def load_all_models(env):
    modellist = [f for f in os.listdir(f"zoo/{env.name}") if f.startswith("_model")]
    modellist.sort()
    models = [load_model(env, "base.zip")]
    for model_name in modellist:
        models.append(load_model(env, name=model_name))
    return models


def get_best_model_name(game_name):
    modellist = [f for f in os.listdir(f"zoo/{game_name}") if f.startswith("_model")]

    if len(modellist) == 0:
        filename = None
    else:
        modellist.sort()
        filename = modellist[-1]

    return filename


def get_model_stats(filename):
    if filename is None:
        generation = 0
        timesteps = 0
        best_rules_based = -np.inf
        best_reward = -np.inf
    else:
        stats = filename.split("_")
        generation = int(stats[2])
        best_rules_based = float(stats[3])
        best_reward = float(stats[4])
        timesteps = int(stats[5])
    return generation, timesteps, best_rules_based, best_reward


def reset_logs(model_directory):
    try:
        filelist = [f for f in os.listdir("logs") if f not in [".gitignore"]]
        for f in filelist:
            if os.path.isfile(f):
                os.remove(f"{"logs"}/{f}")

        for i in range(100):
            if os.path.exists(f"{"logs"}/tb_{i}"):
                rmtree(f"{"logs"}/tb_{i}")

        open(f"{"logs"}/log.txt", "a").close()

    except Exception as e:
        print(e)
        print("Reset logs failed")


def reset_models(model_directory):
    try:
        filelist = [f for f in os.listdir(model_directory) if f not in [".gitignore"]]
        for f in filelist:
            os.remove(f"{model_directory}/{f}")
    except Exception as e:
        print(e)
        print("Reset models failed")
