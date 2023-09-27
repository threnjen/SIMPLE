def get_environment(game_name):
    try:
        if game_name in ("tictactoe"):
            from tictactoe.envs.tictactoe import TicTacToeEnv

            return TicTacToeEnv
        elif game_name in ("connect4"):
            from connect4.envs.connect4 import Connect4Env

            return Connect4Env
        elif game_name in ("sushigo"):
            from sushigo.envs.sushigo import SushiGoEnv

            return SushiGoEnv
        elif game_name in ("butterfly"):
            from butterfly.envs.butterfly import ButterflyEnv

            return ButterflyEnv
        elif game_name in ("geschenkt"):
            from geschenkt.envs.geschenkt import GeschenktEnv

            return GeschenktEnv
        elif game_name in ("frouge"):
            from frouge.envs.frouge import FlammeRougeEnv

            return FlammeRougeEnv
        else:
            raise Exception(f"No environment found for {game_name}")
    except SyntaxError as e:
        print(e)
        raise Exception(f"Syntax Error for {game_name}!")
    except:
        raise Exception(
            f"Install the environment first using: \nbash scripts/install_env.sh {game_name}\nAlso ensure the environment is added to /utils/register.py"
        )


def get_network_arch(game_name):
    if game_name in ("tictactoe"):
        from models.tictactoe.models import CustomPolicy

        return CustomPolicy
    elif game_name in ("connect4"):
        from models.connect4.models import CustomPolicy

        return CustomPolicy
    elif game_name in ("sushigo"):
        from models.sushigo.models import CustomPolicy

        return CustomPolicy
    elif game_name in ("butterfly"):
        from models.butterfly.models import CustomPolicy

        return CustomPolicy
    elif game_name in ("geschenkt"):
        from models.geschenkt.models import CustomPolicy

        return CustomPolicy
    elif game_name in ("frouge"):
        from models.frouge.models import CustomPolicy

        return CustomPolicy
    else:
        raise Exception(f"No model architectures found for {game_name}")
