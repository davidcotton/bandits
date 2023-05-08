from collections import namedtuple


Transition = namedtuple("Transition", ["obs", "action", "next_obs", "reward", "terminal"])
