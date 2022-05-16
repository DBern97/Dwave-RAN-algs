

import numpy as np


class testing:
    def __init__(self):
        # @attribute 
        self.value = 1
    def mul(self):
        return 2*self.value
    def addition(self):
        return self.value + self.mul()

if __name__ == "__main__":

    trial = testing()
    for i in range(5):
        trial.value = i
        print(trial.mul(), trial.addition())
        
    