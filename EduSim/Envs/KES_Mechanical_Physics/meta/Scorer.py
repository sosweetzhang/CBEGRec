import random
from EduSim.Envs.meta import TraitScorer
import numpy as np


class KESASSISTScorer(TraitScorer):
    def response_function(self, user_trait, item_trait, cm=None, *args, **kwargs):
        
        
        if cm:
            return 1 if np.mean(list(cm.values())) >= 0.5 else 0
        return 1 if user_trait[str(item_trait)] >= 0.5 else 0


    def middle_response_function(self, user_trait, item_trait, *args, **kwargs):
        return 1 if random.random() <= user_trait[item_trait] else 0
