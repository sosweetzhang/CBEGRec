
from pprint import pformat
import gym


class Env(gym.Env):
    metadata = {'render.modes': ['human', 'log']}

    def __repr__(self):
        return pformat(self.parameters)

    @property
    def parameters(self) -> dict:
        return {}

    def reset(self):
        raise NotImplementedError

    def render(self, mode='human'):
        return ""

    def step(self, learning_item_id, *args, **kwargs):
        raise NotImplementedError

    def n_step(self, learning_path, *args, **kwargs):
        raise NotImplementedError

    def begin_episode(self, *args, **kwargs):
        raise NotImplementedError

    def end_episode(self, *args, **kwargs):
        raise NotImplementedError
