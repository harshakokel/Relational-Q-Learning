from collections import OrderedDict
from srlearn import Background, Database
from core.srl import RDNRegressor
import numpy as np
from core.data_management import ReplayBuffer
from core.trainer import Trainer
from core.util.logging import logger
from core.util.eval_util import create_stats_ordered_dict
import random
import gtimer as gt
from core.exploration_strategy import EpsilonGreedy

class GBQL(Trainer):
    def __init__(self, n_iter=20, n_trees=5, batch_size=10,
                 train_env=None, bk=None, max_trajectory_length=50,
                 replay_sampling_rate=0.10, test_env=None,
                 max_buffer_size=1000, target_predicate="q_value",
                 learning_rate=0.9, discount_factor=0.99,
                 n_evaluation_trajectories=10, n_burn_in_traj=0,
                 additional_facts=None, goal_q_value=1,
                 exploration_strategy=EpsilonGreedy(), learning_rate_strategy=None,
                 buffer=ReplayBuffer, test_gap=10):
        self.n_iterations = n_iter
        self.n_trees = n_trees
        self.batch_size = batch_size
        self.env = train_env
        self.target = target_predicate
        self.n_estimators = []
        self.test_env = test_env
        if test_env is None:
            self.test_env = train_env
        self.max_traj_len = max_trajectory_length
        self.buffer = buffer(max_size=max_buffer_size)
        self.replay_sampling_rate = replay_sampling_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.n_eval_traj = n_evaluation_trajectories
        self.burn_in_traj = n_burn_in_traj
        self.bk = bk
        if bk is None:
            self.bk = Background()
        self.additional_facts = additional_facts
        self.goal_qvalue = goal_q_value
        self.exploration_strategy = exploration_strategy
        self.learning_rate_strategy = learning_rate_strategy
        self.test_gap = test_gap
        if learning_rate_strategy is not None:
            self.learning_rate = self.learning_rate_strategy.alpha

    def fit_q(self, train, path=None, save=False):
        """Learn a relational Q Function using RDN Boost"""
        bk = self.bk
        reg = RDNRegressor(background=bk, target=self.target, n_estimators=self.n_trees)
        if self.additional_facts is not None:
            train.facts += self.additional_facts
        reg.fit(train, path, preserve_data=save)
        # load cache
        states = self.env.get_state_space()
        if states is not None:
            test = Database()
            for i, state in enumerate(states):
                test.facts += self.env.observe(state, state_id=F"s{i}")
                test.pos += self.env.all_actions(state, state_id=F"s{i}", regression=True)
            if self.additional_facts is not None:
                test.facts += self.additional_facts
            q_values = reg.predict(test)
            q_arr = q_values.reshape((len(states),int(q_values.size/len(states))))
            reg.set_cache(dict(zip(states,q_arr)))
        return reg

    def generate_batch(self, train_batch, batch_size=10, q_function=None):
        """Generate a training batch"""
        new_batch = []
        state_id = 0
        traj_lens = []
        bellman_error = []
        for i in range(batch_size):
            trajectory = []
            done = False
            self.env.reset()
            current_state = self.env.state
            traj_len = 0
            while not done:
                action, q_value = self.get_action(current_state, q_function)
                trajectory.append((current_state, action))
                next_state = self.env.apply_action(action)
                reward = self.env.get_reward(current_state, action, next_state)
                if self.env.is_goal_state(next_state):
                    trajectory.append((next_state, "SUCCESS"))
                    traj_lens.append(traj_len)
                    next_state_qvalue = self.goal_qvalue
                    done = True
                else:
                    _, next_state_qvalue = self.get_action(next_state, q_function, best=True)
                    traj_len += 1
                    if traj_len >= self.max_traj_len - 1:
                        trajectory.append((next_state, "END"))
                        traj_lens.append(traj_len)
                        done = True
                    else:
                        current_state = next_state
                q = ((1.0 - self.learning_rate) * q_value) + (self.learning_rate * (
                        reward + (self.discount_factor * next_state_qvalue)))
                bellman_error.append(abs(q_value - (reward + (self.discount_factor * next_state_qvalue))))
                self.add_sample(train_batch, current_state, action, q, state_id=f"n{state_id}")
                state_id += 1
            new_batch.append(trajectory)
        return new_batch, state_id, traj_lens, bellman_error

    def get_training_batch(self, batch_size, q_function):
        replay_traj = []
        train_batch = Database()
        if self.buffer.size > 0:
            # Sample historic trajectories
            replay_traj = self.buffer.get_trajectories(int(self.replay_sampling_rate * batch_size))
        gt.stamp('sampled historic trajectories', unique=False)
        new_traj, env_steps, traj_lens, bellman_error = self.generate_batch(train_batch, batch_size - len(replay_traj),
                                                                           q_function)
        state_id = env_steps
        gt.stamp('sampled new trajectories', unique=False)
        self.buffer.add_all_trajectories(new_traj)
        for traj in replay_traj:
            # For all states in trajectory except last
            for i in range(len(traj) - 1):
                current_state, action, next_state = traj[i][0], traj[i][1], traj[i + 1][0]
                reward = self.env.get_reward(current_state, action, next_state)
                q_value = self.get_qvalue(current_state, action, q_function)
                if traj[i+1][1] == "SUCCESS":
                    next_q_value = self.goal_qvalue
                else:
                     _, next_q_value = self.get_action(next_state, q_function, best=True)
                q = ((1.0 - self.learning_rate) * q_value) + self.learning_rate * (
                        reward + (self.discount_factor * next_q_value))
                bellman_error.append(abs(q_value - (reward + (self.discount_factor * next_q_value))))
                self.add_sample(train_batch, current_state, action, q, state_id=str(state_id))
                state_id += 1
        gt.stamp('evaluate historic trajectories', unique=False)
        stats = create_stats_ordered_dict(
            'bellman error',
            bellman_error,
        )
        stats['batch size'] = batch_size
        stats['learning rate'] = self.learning_rate
        stats['discount factor'] = self.discount_factor
        stats['steps in env'] = env_steps
        stats['sample size'] = state_id
        stats['no of replay traj'] = len(replay_traj)
        stats['no of sampled traj'] = len(new_traj)
        return train_batch, stats

    def get_qvalue(self, state, action, q_function=None, env=None):
        if env is None:
            env = self.env
        if q_function is None:
            return 0.0
        possible_actions = self.env.all_actions(state)
        if action not in possible_actions:
            raise Exception("Invalid action")
        _, q_values, _ = predict(env, q_function, state, self.additional_facts)
        idx = possible_actions.index(action)
        return q_values[idx]

    def get_action(self, state, q_function=None, env=None, best=False):
        if env is None:
            env = self.env
        possible_actions = env.all_actions(state)
        if q_function is None:
            action = random.choice(possible_actions)
            return action, 0.0
        idx, q_values, best_action = predict(env, q_function, state, self.additional_facts)
        if not best:
            idx = self.exploration_strategy.get_action_idx(idx, len(possible_actions))
        return possible_actions[idx], q_values[idx]

    def train(self):
        """Fitted Q Learning"""
        current_q = None
        if self.burn_in_traj > 0:
            logger.log(f"adding {self.burn_in_traj} burn_in_traj")
            train_batch = Database()
            traj, _, _, _ = self.generate_batch(train_batch, self.burn_in_traj)
            self.buffer.add_all_trajectories(traj)
        logger.log("started fitted Q training")
        for i in gt.timed_for(range(self.n_iterations), save_itrs=True):
            logger.log(f"Iteration {i} started")
            logger.log(f"Iteration {i} getting training batch")
            train_batch, training_stats = self.get_training_batch(self.batch_size, current_q)
            gt.stamp("training batch", unique=False)
            logger.log(f"Iteration {i} fitting q function")
            updated_q = self.fit_q(train_batch, path=f"{logger.get_snapshot_dir()}/fitted-q/itr{i}", save=True)
            gt.stamp("bsrl learning", unique=False)
            self.n_estimators.append(updated_q)
            if i % self.test_gap == 0:
                logger.log(f"Iteration {i} evaluating")
                paths = self.evaluate(self.n_eval_traj, updated_q)
                gt.stamp("bsrl evaluation", unique=False)
                self._log_stat(updated_q, training_stats, paths, i)
                logger.record_dict(self.exploration_strategy.stats(), prefix='exploration/')
                logger.dump_tabular()
                self.exploration_strategy.end_epoch()
                if self.learning_rate_strategy is not None:
                    self.learning_rate = self.learning_rate_strategy.end_epoch()
            current_q = updated_q
            logger.log(f"Iteration {i} ended")
        return current_q

    def _log_stat(self, q_function, training_stats, paths, itr):
        logger.save_itr_params(itr, q_function)
        logger.record_dict(training_stats, prefix='training/')
        buffer_stats = self.buffer.get_diagnostics()
        logger.record_dict(buffer_stats, prefix='buffer/')
        evaluation_stats = self.env.get_diagnostics(paths)
        logger.record_dict(evaluation_stats, prefix='evaluation/')
        logger.save_eval_data(paths, itr=itr)
        logger.record_tabular('iteration', itr)
        times_itrs = gt.get_times().stamps.itrs
        times = OrderedDict()
        epoch_time = 0
        for key in sorted(times_itrs):
            time = times_itrs[key][-1]
            epoch_time += time
            times['{} (s)'.format(key)] = time
        times['iteration (s)'] = epoch_time
        times['total (s)'] = gt.get_times().total
        logger.record_dict(times, prefix=f'time/')

    def add_sample(self, train_batch, state, action, q_value, state_id='0'):
        train_batch.facts += self.env.observe(state, state_id=f"s{state_id}")
        action = action.replace("_", f"s{state_id}")
        train_batch.pos.append(f"regressionExample({action},{q_value:.3f}).")
        return

    def evaluate(self, batch_size, q_function):
        """Evaluation in Test env """
        paths = []
        # TODO: Evaluation takes maximum time, this could be improved with parallel environments
        for i in range(batch_size):
            path = dict(states=[], actions=[], rewards=[], info=[],
                        episode_reward=0, is_success=False, episode_length=0, percent_solved=0.0)
            done = False
            self.test_env.reset()
            current_state = self.test_env.state
            traj_len = 0
            while not done:
                action, _ = self.get_action(current_state, q_function, env=self.test_env, best=True)
                path['states'].append(current_state)
                path['actions'].append(action)
                next_state = self.test_env.apply_action(action)
                r = self.test_env.get_reward(current_state, action, next_state)
                path['rewards'].append(r)
                path['episode_reward'] += r
                # if current_state == next_state:
                #     traj_len = self.max_traj_len
                traj_len += 1
                solved = self.test_env.is_goal_state(next_state)
                if solved or traj_len >= self.max_traj_len:
                    path['states'].append(next_state)
                    path['is_success'] = solved
                    path['episode_length'] = traj_len
                    path['percent_solved'] = self.test_env.get_solved_percent()
                    done = True
                else:
                    current_state = next_state
            paths.append(path)
        return paths


def predict(env, q_function, state, additional_facts=None):
    # check cache
    q_values = q_function.fetch_cache(state)
    if q_values is not None:
        all_actions = env.all_actions(state, state_id="s1", regression=True)
        where_max = np.where(q_values == np.max(q_values))[0]
        if len(where_max) == 1:
            idx = where_max[0]
        else:
            idx = np.random.choice(where_max)
        return idx, q_values, all_actions[idx]
    test = Database()
    test.facts = env.observe(state, state_id="s1")
    test.pos = env.all_actions(state, state_id="s1", regression=True)
    if additional_facts is not None:
        test.facts += additional_facts
    q_values = q_function.predict(test)
    where_max = np.where(q_values == np.max(q_values))[0]
    if len(where_max) == 1:
        idx = where_max[0]
    else:
        idx = np.random.choice(where_max)
    return idx, q_values, test.pos[idx]


class RRT(GBQL):
    """The RRT code is same as GBQL but with only 1 tree"""

    def __init__(self, n_iter=1, batch_size=10, train_env=None, bk=None, max_trajectory_length=50,
                 replay_sampling_rate=0.10, test_env=None, max_buffer_size=1000, target_predicate="q_value",
                 learning_rate=0.9, discount_factor=0.99, n_evaluation_trajectories=10,
                 n_burn_in_traj=0, additional_facts=None, goal_q_value=1, exploration_strategy=EpsilonGreedy(),
                 learning_rate_strategy=None, buffer=ReplayBuffer, test_gap=10):
        super().__init__(n_iter=n_iter, n_trees=1, batch_size=batch_size, train_env=train_env, bk=bk,
                         max_trajectory_length=max_trajectory_length, replay_sampling_rate=replay_sampling_rate,
                         test_env=test_env, max_buffer_size=max_buffer_size, target_predicate=target_predicate,
                         learning_rate=learning_rate, discount_factor=discount_factor,
                         n_evaluation_trajectories=n_evaluation_trajectories, n_burn_in_traj=n_burn_in_traj,
                         additional_facts=additional_facts, goal_q_value=goal_q_value,
                         exploration_strategy=exploration_strategy, learning_rate_strategy=learning_rate_strategy,
                         buffer=buffer, test_gap=test_gap)


