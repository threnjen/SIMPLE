import random

INPUT_CONFIG = {
    "reset": [False, bool, "Start retraining the model from scratch"],
    "opponent_type": [
        "mostly_best",
        str,
        "best / mostly_best / random / base / rules - the type of opponent to train against",
    ],
    "debug": [False, bool, "Debug logging"],
    "verbose": [False, bool, "Show observation in debug output"],
    "rules": [False, bool, "Evaluate on a ruled-based agent"],
    "best": [False, bool, "Uses best moves when evaluating agent against rules-based agent"],
    "game_name": [
        "tictactoe",
        str,
        "Which gym environment to train in: tictactoe, connect4, sushigo, butterfly, geschenkt, frouge",
    ],
    "seed": [random.randint(0, 1000), int, "Random seed"],
    "eval_freq": [10240, int, "How many timesteps should each actor contribute before the agent is evaluated?"],
    "n_eval_episodes": [100, int, "How many episodes should each actor contirbute to the evaluation of the agent"],
    "threshold": [0.2, float, "What score must the agent achieve during evaluation to 'beat' the previous version?"],
    "gamma": [0.99, float, "The value of gamma in PPO"],
    "timesteps_per_actorbatch": [1024, int, "How many timesteps should each actor contribute to the batch?"],
    "clip_param": [0.2, float, "The clip paramater in PPO"],
    "entcoeff": [0.1, float, "The entropy coefficient in PPO"],
    "optim_epochs": [4, int, "The number of epoch to train the PPO agent per batch"],
    "optim_stepsize": [0.0003, float, "The step size for the PPO optimiser"],
    "optim_batchsize": [1024, int, "The minibatch size in the PPO optimiser"],
    "lam": [0.95, float, "The value of lambda in PPO"],
    "adam_epsilon": [1e-05, float, "The value of epsilon in the Adam optimiser"],
}
