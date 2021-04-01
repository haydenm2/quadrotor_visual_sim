import numpy as np


class targets:
    def __init__(self):
        target_1 = np.array([1,   1, 0])
        target_2 = np.array([1,  -1, 0])
        target_3 = np.array([-1, -1, 0])
        self.targets = np.column_stack((target_1, target_2, target_3))

    def get_targets(self):
        return self.targets
