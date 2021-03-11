from core.trainer import RRT
from core.exploration_strategy import EpsilonGreedyWithExponentialDecay
from environments.blocksworld_stack import BlocksWorldStack
from core.util.launcher_util import setup_logger
import gtimer as gt
from srlearn import Background

variant = {
    'trainer': 'rrt',
    'env_kwargs': {'all_blocks_on_floor': True,
                   'intermittent_reward': False,
                   'terminal_reward': 1.2,
                   'step_cost': -0.2,
                   'n_blocks_options': [3, 4, 5]},
    'bk_kwargs': {'max_tree_depth': 6,
                   'node_size': 2,
                  'ok_if_unknown':  ["heightlessthan/3."]
                  },
    'trainer_kwargs': {'n_iter': 100,
                       'max_buffer_size': 500,
                       'target_predicate': 'move',
                       'learning_rate':0.1,
                       'test_gap':1},
    'exploration_strategy': EpsilonGreedyWithExponentialDecay,
    'test_env_kwargs': {'all_blocks_on_floor': True,
                        'intermittent_reward': False,
                        'terminal_reward': 1.2,
                        'step_cost': -0.2,
                        'n_blocks_options': [6, 7]},
    'modes':["clear(+state,+block).",
             "clear(+state,-block).",
             "isFloor(+block).",
             "heightlessthan(+state,+block,+block).",
             # "on(+state,+block,-block).",
             # "on(+state,-block,+block).",
             "move(+state,+block,+block)."]}

n_iter = 5
for i in range(n_iter):
    variant['experiment_no'] = i
    setup_logger(f"{variant['trainer']}-stack", variant=variant, snapshot_mode="all", exp_id=i)
    train_env = BlocksWorldStack(**variant['env_kwargs'])
    bk = Background(modes=variant['modes'], **variant['bk_kwargs'])
    test_env = BlocksWorldStack(**variant['test_env_kwargs'])
    RRT_Trainer = RRT(train_env=train_env, bk=bk, test_env=test_env, exploration_strategy=variant['exploration_strategy'](),
                      **variant['trainer_kwargs'])
    fitted_q = RRT_Trainer.train()
    gt.reset_root()
