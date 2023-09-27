# docker-compose exec app python3 train.py -r -e butterfly

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

tf.get_logger().setLevel("INFO")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


import argparse
import time
from shutil import copyfile
from mpi4py import MPI
from arguments import INPUT_CONFIG

from stable_baselines.ppo1 import PPO1
from stable_baselines.common.callbacks import EvalCallback

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import logger

from utils.callbacks import SelfPlayCallback
from utils.files import reset_logs, reset_models
from utils.register import get_network_arch, get_environment
from utils.selfplay import selfplay_wrapper

import config

model_reset = INPUT_CONFIG["reset"][0]
opponent_type = INPUT_CONFIG["opponent_type"][0]
debug_logging = INPUT_CONFIG["debug"][0]
verbose_output = INPUT_CONFIG["verbose"][0]
use_rules = INPUT_CONFIG["rules"][0]
use_best = INPUT_CONFIG["best"][0]
game_name = INPUT_CONFIG["game_name"][0]
random_seed = INPUT_CONFIG["seed"][0]
eval_freq = INPUT_CONFIG["eval_freq"][0]
n_eval_episodes = INPUT_CONFIG["n_eval_episodes"][0]
threshold = INPUT_CONFIG["threshold"][0]
gamma = INPUT_CONFIG["gamma"][0]
timesteps_per_actorbatch = INPUT_CONFIG["timesteps_per_actorbatch"][0]
clip_param = INPUT_CONFIG["clip_param"][0]
entcoeff = INPUT_CONFIG["entcoeff"][0]
optim_epochs = INPUT_CONFIG["optim_epochs"][0]
optim_stepsize = INPUT_CONFIG["optim_stepsize"][0]
optim_batchsize = INPUT_CONFIG["optim_batchsize"][0]
lam = INPUT_CONFIG["lam"][0]
adam_epsilon = INPUT_CONFIG["adam_epsilon"][0]


def main(arg_config):
    rank = MPI.COMM_WORLD.Get_rank()

    model_directory = f"zoo/{game_name}"

    if rank == 0:
        try:
            os.makedirs(model_directory)
        except:
            pass
        reset_logs(model_directory)
        if model_reset:
            reset_models(model_directory)
        logger.configure("logs")
    else:
        logger.configure(format_strs=[])

    if debug_logging:
        logger.set_level(config.DEBUG)
    else:
        time.sleep(5)
        logger.set_level(config.INFO)

    workerseed = random_seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    logger.info("\nSetting up the selfplay training environment opponents...")
    game_env = get_environment(game_name)
    env = selfplay_wrapper(game_env)(opponent_type=opponent_type, verbose=verbose_output)
    env.seed(workerseed)

    CustomPolicy = get_network_arch(game_name)

    params = {
        "gamma": gamma,
        "timesteps_per_actorbatch": timesteps_per_actorbatch,
        "clip_param": clip_param,
        "entcoeff": entcoeff,
        "optim_epochs": optim_epochs,
        "optim_stepsize": optim_stepsize,
        "optim_batchsize": optim_batchsize,
        "lam": lam,
        "adam_epsilon": adam_epsilon,
        "schedule": "linear",
        "verbose": 1,
        "tensorboard_log": "logs",
    }

    time.sleep(5)  # allow time for the base model to be saved out when the environment is created

    if model_reset or not os.path.exists(f"{model_directory}/best_model.zip"):
        logger.info("\nLoading the base PPO agent to train...")
        model = PPO1.load(f"{model_directory}/base.zip", env, **params)
    else:
        logger.info("\nLoading the best_model.zip PPO agent to continue training...")
        model = PPO1.load(f"{model_directory}/best_model.zip", env, **params)

    # Callbacks
    logger.info("\nSetting up the selfplay evaluation environment opponents...")
    callback_args = {
        "eval_env": selfplay_wrapper(game_env)(opponent_type=opponent_type, verbose=verbose_output),
        "best_model_save_path": "zoo/tmp",
        "log_path": "logs",
        "eval_freq": eval_freq,
        "n_eval_episodes": n_eval_episodes,
        "deterministic": False,
        "render": True,
        "verbose": 0,
    }

    if use_rules:
        logger.info("\nSetting up the evaluation environment against the rules-based agent...")
        # Evaluate against a 'rules' agent as well
        eval_actual_callback = EvalCallback(
            eval_env=selfplay_wrapper(game_env)(opponent_type="rules", verbose=verbose_output),
            eval_freq=1,
            n_eval_episodes=n_eval_episodes,
            deterministic=use_best,
            render=True,
            verbose=0,
        )
        callback_args["callback_on_new_best"] = eval_actual_callback

    # Evaluate the agent against previous versions
    eval_callback = SelfPlayCallback(opponent_type, threshold, game_name, **callback_args)

    logger.info("\nSetup complete - commencing learning...\n")

    model.learn(total_timesteps=int(1e9), callback=[eval_callback], reset_num_timesteps=False, tb_log_name="tb")

    env.close()
    del env


if __name__ == "__main__":
    main(INPUT_CONFIG)
