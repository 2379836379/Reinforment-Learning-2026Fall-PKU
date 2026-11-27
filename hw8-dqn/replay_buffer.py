"""Uniform and proportional prioritized replay for FrameNumpy transitions."""
import random

import numpy as np

from sample import SampleBatchNumpy


class SumMinTree:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError('capacity must be positive')
        self.capacity = capacity
        self.leaves = 1 << (capacity - 1).bit_length()
        self.sums = np.zeros(2 * self.leaves, dtype=np.float64)
        self.mins = np.full(2 * self.leaves, np.inf, dtype=np.float64)

    def update(self, index, value):
        node = self.leaves + index
        self.sums[node] = value
        self.mins[node] = value if value > 0 else np.inf
        node //= 2
        while node:
            self.sums[node] = self.sums[node * 2] + self.sums[node * 2 + 1]
            self.mins[node] = min(self.mins[node * 2], self.mins[node * 2 + 1])
            node //= 2

    def find_prefixsum(self, value):
        if not 0 <= value < self.sums[1]:
            raise ValueError('prefix sum is outside the tree total')
        node = 1
        while node < self.leaves:
            left = node * 2
            if value < self.sums[left]:
                node = left
            else:
                value -= self.sums[left]
                node = left + 1
        index = node - self.leaves
        if index >= self.capacity:
            raise RuntimeError('sum tree selected an unused leaf')
        return index


class PrioritizedReplayBuffer:
    """Proportional PER with stratified sampling and normalized IS weights."""
    def __init__(self, capacity, alpha=.6, priority_epsilon=1e-6):
        if not 0 <= alpha <= 1 or priority_epsilon <= 0:
            raise ValueError('alpha must be in [0, 1] and priority_epsilon positive')
        self.capacity = capacity
        self.alpha = alpha
        self.priority_epsilon = priority_epsilon
        self.storage = [None] * capacity
        self.tree = SumMinTree(capacity)
        self.next_index = 0
        self.size = 0
        self.max_priority = 1.0

    def __len__(self):
        return self.size

    def append(self, transition):
        index = self.next_index
        self.storage[index] = transition
        self.tree.update(index, self.max_priority ** self.alpha)
        self.next_index = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def extend(self, transitions):
        for transition in transitions:
            self.append(transition)

    def sample(self, batch_size, beta):
        if self.size < batch_size or batch_size < 1:
            raise ValueError('not enough transitions to sample the requested batch')
        if not 0 <= beta <= 1:
            raise ValueError('beta must be in [0, 1]')
        total = self.tree.sums[1]
        segment = total / batch_size
        indices = [self.tree.find_prefixsum((i + random.random()) * segment)
                   for i in range(batch_size)]
        probabilities = np.asarray([self.tree.sums[self.tree.leaves + i] / total for i in indices])
        min_probability = self.tree.mins[1] / total
        max_weight = (self.size * min_probability) ** (-beta)
        weights = ((self.size * probabilities) ** (-beta) / max_weight).astype(np.float32)
        batch = SampleBatchNumpy.stack([self.storage[index] for index in indices])
        batch.weights = weights.reshape(-1, 1)
        return batch, np.asarray(indices, dtype=np.int64)

    def update_priorities(self, indices, priorities):
        indices = np.asarray(indices).reshape(-1)
        priorities = np.asarray(priorities).reshape(-1)
        if len(indices) != len(priorities):
            raise ValueError('indices and priorities must have equal lengths')
        for index, priority in zip(indices, priorities):
            if not 0 <= index < self.size or not np.isfinite(priority) or priority < 0:
                raise ValueError('invalid replay index or priority')
            priority = float(priority) + self.priority_epsilon
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(int(index), priority ** self.alpha)


class UniformReplayBuffer:
    """Small API-compatible wrapper around the prior deque implementation."""
    def __init__(self, capacity):
        from collections import deque
        self.storage = deque(maxlen=capacity)

    def __len__(self):
        return len(self.storage)

    def append(self, transition):
        self.storage.append(transition)

    def extend(self, transitions):
        self.storage.extend(transitions)

    def sample(self, batch_size, beta=1.0):
        del beta
        indices = random.sample(range(len(self.storage)), batch_size)
        batch = SampleBatchNumpy.stack([self.storage[index] for index in indices])
        batch.weights = np.ones((len(indices), 1), dtype=np.float32)
        return batch, np.asarray(indices, dtype=np.int64)

    def update_priorities(self, indices, priorities):
        del indices, priorities
