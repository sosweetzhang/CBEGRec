
__all__ = ["irt"]

import numpy as np
import math

import torch


def irt(ability, difficulty, discrimination=5, c=0.25):
    return c + (1 - c) / (1 + math.exp(-1.7 * discrimination * (ability - difficulty)))


def dina(abilities, guessing, skipping):
    eta = np.prod(abilities)
    return guessing ** (1 - eta) * (1 - skipping) ** eta
