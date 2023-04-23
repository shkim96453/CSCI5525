#MyThompsonSampling
import numpy as np

class MyThompsonSampling:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.success = np.ones(num_arms)
        self.failure = np.ones(num_arms)

    def pull_arm(self):
        theta = np.random.beta(self.success, self.failure)
        arm = np.argmax(theta)
        self.arm = arm 
        return self.arm

    def update_model(self, reward):
        if reward == 1:
            self.success[self.arm] += 1

        else: 
            self.failure[self.arm] += 1