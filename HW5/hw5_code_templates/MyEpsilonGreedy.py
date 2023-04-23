# Epsilln Greedy
import numpy as np

class MyEpsilonGreedy:
    def __init__(self, num_arms, epsilon):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.values = np.zeros(num_arms)
        self.counts = np.zeros(num_arms)

    def pull_arm(self):
        if np.random.random() < self.epsilon:
            arm = np.random.choice(self.num_arms)
        else:
            arm = np.argmax(self.values)
        
        self.counts[arm] += 1
        self.arm = arm
        return self.arm

    def update_model(self, reward):
        self.values[self.arm] = (self.values[self.arm]*self.counts[self.arm] + reward) / (self.counts[self.arm]+1)
        self.counts[self.arm] += 1