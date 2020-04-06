import random
from utils.Utils import get_transition

" Module of memory "
" Written by S.Romero 3/2020 "
" Taken from NN Pytorch tutorial"

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.Transition = get_transition()

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Samples a transition from storage"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
