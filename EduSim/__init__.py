
from gym.envs.registration import register
from .SimOS import train_eval, MetaAgent
from .spaces import *
from .Envs import KESPhysicsEnv

register(
    id='KESMP-v1',
    entry_point='EduSim.Envs:KESPhysicsEnv',
)
