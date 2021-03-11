from environments import Environment, State, Task
import numpy as np
import re
from copy import deepcopy

FLOOR = "floor"

TOP = "clear"
IS_FLOOR = "isFloor"
ON = "on"
MOVE = "move"
HEIGHT_LT = "heightlessthan"

regexPattern = '|'.join(map(re.escape, ["(", ")", ","]))
pattern = re.compile(regexPattern)


class BlocksWorldState(State):

    def __init__(self, n_blocks=4, n_towers=1, env_id=1, r=None):
        self.n_blocks = n_blocks
        rand = np.random.random(n_towers)
        if r is None:
            r = ((rand * (n_blocks - n_towers)) / rand.sum()).astype(int)
            j = 0
            while r.sum() < (n_blocks - n_towers):
                r[j] += 1
                j += 1
        self.towers = [[F"b{j + r[:i].sum() + i}" for j in range(r[i] + 1)] for i in range(len(r))]
        self.tower_hash = [[F"b_{j + r[:i].sum() + i}" for j in range(r[i] + 1)] for i in range(len(r))]

    def __eq__(self, other):
        return self.towers == other.towers

    def __str__(self):
        return self.towers.__str__()

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__str__())


class BlocksWorldStack(Environment, Task):

    def env_name(self):
        return "BlocksWorld-Stack"

    def __init__(self, env_id=1, n_blocks=3, n_blocks_options=None, all_blocks_on_floor=True,
                 intermittent_reward=False, terminal_reward=1, step_cost=-0.2, add_height_less_than=True):
        """

        :type all_blocks_on_floor: bool
        """
        self.n_blocks = n_blocks
        # self.n_towers = n_towers
        self.all_blocks_on_floor = all_blocks_on_floor
        self.n_blocks_options = n_blocks_options
        self.add_height_less_than = add_height_less_than
        self.env_id = env_id
        if self.n_blocks_options is not None:
            self.n_blocks = self.n_blocks_options[self.env_id % len(self.n_blocks_options)]
        if self.all_blocks_on_floor:
            self.n_towers = self.n_blocks
        self.step_cost = step_cost
        self.intermittent_reward = intermittent_reward
        self.terminal_reward = terminal_reward
        # assert n_towers < n_blocks, "More tower than blocks"
        super().__init__(BlocksWorldState(self.n_blocks, self.n_towers, env_id))

    def get_reward(self, state, action, next_state=None):
        """
        :type state: :class:`environments.blocksworld_stack.BlocksWorldState`
        :type action: str
                predicate action
        :type next_state: :class:`environments.blocksworld_stack.BlocksWorldState`
        """
        reward = self.step_cost
        if next_state is None:
            if action == 'END':
                return reward
            elif action == "SUCCESS":
                return reward + self.terminal_reward
            else:
                raise NotImplementedError
        if not self.is_goal_state(state) and self.is_goal_state(next_state):
            reward += self.terminal_reward
        else:
            if self.intermittent_reward:
                max_height = 0
                for tower in next_state.towers:
                    if max_height < len(tower):
                        max_height = len(tower)
                reward -= (1.0 / max_height)
        return reward

    @property
    def state(self):
        return deepcopy(self._state)

    def apply_action(self, a):
        # Hack
        action = a.replace("regressionExample(", "")
        action = pattern.split(action)
        from_tower = None
        for t in range(len(self._state.towers)):
            if self._state.towers[t][-1] == action[2]:
                from_tower = t
                break
        if from_tower is None:
            # raise Exception("Invalid action selected")
            return deepcopy(self._state)
        if action[3] == FLOOR:
            # Already on floor
            if len(self._state.towers[from_tower]) == 1:
                return deepcopy(self._state)
            block = self._state.towers[from_tower].pop()
            self._state.towers.append([block])
        else:
            to_tower = None
            for t in range(len(self._state.towers)):
                if self._state.towers[t][-1] == action[3]:
                    to_tower = t
                    break
            if to_tower is None:
                # raise Exception("Invalid action selected")
                return deepcopy(self._state)
            block = self._state.towers[from_tower].pop()
            self._state.towers[to_tower].append(block)
        if not self._state.towers[from_tower]:
            self._state.towers.remove([])
        return deepcopy(self._state)

    def get_solved_percent(self, state=None):
        if state is None:
            state = self._state
        max_height = 0
        for tower in state.towers:
            if max_height < len(tower):
                max_height = len(tower)
        return max_height / state.n_blocks

    def all_actions(self, state=None, state_id="_", regression=False):
        if state is None:
            state = self._state
        possible_actions = []
        predicate = "{MOVE}({state_id},{i},{j})"
        if regression:
            predicate = "regressionExample({MOVE}({state_id},{i},{j}),0.0)."
        for from_tower in state.towers:
            if len(from_tower) > 1 and len(state.towers) > 1:
                possible_actions.append(predicate.format(MOVE=MOVE,state_id=state_id, i=from_tower[-1], j=FLOOR))
            for i in from_tower:
                for to_tower in state.towers:
                    for j in to_tower:
                        possible_actions.append(predicate.format(MOVE=MOVE,state_id=state_id, i=i, j=j))
        return possible_actions

    @staticmethod
    def all_valid_actions(possible_actions, predicate, state, state_id):
        top_most_blocks = []
        for tower in state.towers:
            top_most_blocks.append(tower[-1])
        for i in top_most_blocks:
            possible_actions.append(predicate.format(MOVE=MOVE,state_id=state_id, i=i, j=FLOOR))
            for j in top_most_blocks:
                if i == j:
                    continue
                possible_actions.append(predicate.format(state_id=state_id, i=i, j=j))

    def cost(self):
        return -0.2

    def observe(self, state=None, state_id="_"):
        if state is None:
            state = self._state
        facts = [f"{IS_FLOOR}({FLOOR})."]
        for tower in state.towers:
            facts.append(f"{ON}({state_id},{tower[0]},{FLOOR}).")
            for i in range(len(tower) - 1):
                facts.append(f"{ON}({state_id},{tower[i + 1]},{tower[i]}).")
            facts.append(f"{TOP}({state_id},{tower[-1]}).")
            if self.add_height_less_than:
                for other_tower in state.towers:
                    if tower == other_tower:
                        continue
                    if len(tower) <len(other_tower):
                        for lb in tower:
                            for hb in other_tower:
                                facts.append(f"{HEIGHT_LT}({state_id},{lb},{hb}).")
        return facts

    def reset(self, state=None):
        if state is not None:
            self._state = state
            return
        self.env_id += 1
        if self.n_blocks_options is not None:
            self.n_blocks = self.n_blocks_options[self.env_id % len(self.n_blocks_options)]
        if self.all_blocks_on_floor:
            self.n_towers = self.n_blocks
        super().__init__(BlocksWorldState(self.n_blocks, self.n_towers, self.env_id))

    @property
    def modes(self):
        """Return mode for the domain"""
        return [f"{TOP}(+state,+block).",
                f"{TOP}(+state,-block).",
                f"{IS_FLOOR}(+block).",
                f"{ON}(+state,+block,-block).",
                f"{ON}(+state,-block,+block).",
                f"{HEIGHT_LT}(+state,+block,+block).",
                f"{MOVE}(+state,+block,+block)."]

    @property
    def target(self):
        """Return target predicate name for the domain"""
        return "move"

    def is_goal_state(self, state=None):
        if state is None:
            state = self._state
        for tower in state.towers:
            if len(tower) == state.n_blocks:
                return True
        return False

    # @staticmethod
    # def get_state_space():
    #     state_space = []
    #     n_blocks = [2,3,4,5,6,7]
    #     kwargs = {2:[(2,[1,1])],
    #               3:[(3,[1,1,1]),(2,[2,1]),(2,[1,2])],
    #               4:[(4,[1,1,1,1]),(3,[2,1,1]),(3,[1,2,1]),(3,[1,1,2]),
    #                  (2,[3,1]),(2,[1,3]),(2,[2,2])],
    #               5:[(5,[1,1,1,1,1]),(4,[2,1,1,1]),(4,[1,2,1,1]),(4,[1,1,2,1]),(4,[1,1,1,2]),
    #                  (3,[3,1,1]),(3,[1,3,1]),(3,[1,1,3]),(3,[2,2,1]),(3,[2,1,2]),(3,[1,2,2]),
    #                  (2,[4,1]),(2,[1,4]),(2,[3,2]),(2,[2,3])],
    #               6:[(6,[1,1,1,1,1,1]),(5,[2,1,1,1,1]),(5,[1,2,1,1,1]),(5,[1,1,2,1,1]),(5,[1,1,1,2,1]),(5,[1,1,1,1,2]),
    #                  (4,[3,1,1,1]),(4,[1,3,1,1]),(4,[1,1,3,1]),(4,[2,2,1,1]),(4,[2,1,2,1]),(4,[2,1,1,2]),(4,[1,2,2,1]),
    #                  (4,[1,2,1,2]),(4,[1,1,2,2]),
    #                  (3,[4,1,1]),(3,[1,4,1]),(3,[1,1,4]),(3,[2,2,2]),(3,[3,1,2]),(3,[3,2,1]),(3,[1,3,2]),(3,[2,3,1]),
    #                  (3,[1,2,3]),(3,[2,1,3]),
    #                  (2,[5,1]),(2,[1,5]),(2,[4,2]),(2,[2,4]),(2,[3,3])],
    #               7:[(7,[1,1,1,1,1,1,1]),(6,[2,1,1,1,1,1]),(6,[1,2,1,1,1,1]),(6,[1,1,2,1,1,1]),(6,[1,1,1,2,1,1]),
    #                  (6,[1,1,1,1,2,1]),(6,[1,1,1,1,1,2]),
    #                  (5,[2,2,1,1,1]),(5,[1,2,2,1,1]),(5,[1,1,2,2,1]),(5,[1,1,1,2,2]),(5,[2,1,1,1,2]),
    #                  (5,[2,1,2,1,1]),(5,[2,1,1,2,1]),(5,[1,2,1,2,1]),(5,[1,2,1,1,2]),(5,[1,1,2,1,2]),
    #                  (5,[3,1,1,1,1]),(5,[1,3,1,1,1]),(5,[1,1,1,3,1]),(5,[1,1,1,1,3]),(5,[1,1,3,1,1]),
    #                  (4,[4,1,1,1]),(4,[1,4,1,1]),(4,[1,1,4,1]),
    #                  (4,[3,2,1,1]),(4,[3,1,2,1]),(4,[3,1,1,2]),(4,[1,3,2,1]),(4,[1,3,1,2]),(4,[1,1,3,2]),
    #                  (4,[2,3,1,1]),(4,[2,1,3,1]),(4,[2,1,1,3]),(4,[1,2,3,1]),(4,[1,2,1,3]),(4,[1,1,2,3]),
    #                  (3,[5,1,1]),(3,[1,5,1]),(3,[1,1,5]),(3,[3,2,2]),(3,[2,3,2]),(3,[2,2,3]),
    #                  (3,[4,1,2]),(3,[4,2,1]),(3,[1,4,2]),(3,[2,4,1]),(3,[1,2,4]),(3,[2,1,4]),
    #                  (3,[3,1,3]),(3,[3,3,1]),(3,[1,3,3]),(3,[2,2,3]),(3,[2,3,2]),(3,[3,2,2]),
    #                  (2,[6,1]),(2,[1,6]),(2,[5,2]),(2,[2,5]),(2,[4,3]),(2,[3,4])],
    #               }
    #     # n_towers = [2,3,4]
    #     for blk in n_blocks:
    #         for comb in kwargs[blk]:
    #            state_space.append(BlocksWorldState(n_blocks, comb[0], 1, r=np.array(comb[1])))
    #     return state_space

if __name__ == '__main__':
    test_env_kwargs= {'all_blocks_on_floor': True,
                       'intermittent_reward': False,
                       'terminal_reward': 1.2,
                       'step_cost': -0.2,
                       'n_blocks_options': [3, 4, 5]}
    env = BlocksWorldStack(**test_env_kwargs)
    state_space = env.get_state_space()

    # env = BlocksWorldStack(n_blocks=10, n_towers=4, n_blocks_options=[3, 4, 5])
    state = env.state
    print(state)
    print(env.observe())
    actions = env.all_actions(regression=False)
    print(actions)
    next_state = env.apply_action(actions[5])
    print(next_state)
    reward = env.get_reward(state, actions[5], next_state)
    print(reward)
