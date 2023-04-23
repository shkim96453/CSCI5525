# MyUCB
import numpy as np

class MyUCB:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.values = np.zeros(num_arms)
        self.counts = np.zeros(num_arms)
        self.total_counts = 0

    def pull_arm(self):
        if self.num_arms > self.total_counts:
            pre_arms = np.arange(0, self.num_arms)
            arm = pre_arms[self.total_counts]
            self.counts[arm] += 1
            self.total_counts += 1
            self.arm = arm
            return self.arm
        else:
            upper_ci = self.values + np.sqrt(2*np.log(self.total_counts)/(self.counts))
            arm = np.argmax(upper_ci)
            self.counts[arm] += 1
            self.total_counts += 1
            self.arm = arm
            return self.arm

    def update_model(self, reward):
        self.values[self.arm] = (self.values[self.arm]*self.counts[self.arm] + reward) / (self.counts[self.arm]+1)
        self.counts[self.arm] += 1