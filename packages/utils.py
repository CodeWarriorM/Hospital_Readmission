import numpy as np

# Classification of diagnose 1
def classify_diag_level1(value):
    if value >= 390 and value < 460 or np.floor(value) == 785:
        return 1
    elif value >= 460 and value < 520 or np.floor(value) == 786:
        return 2
    elif value >= 520 and value < 580 or np.floor(value) == 787:
        return 3
    elif np.floor(value) == 250:
        return 4
    elif value >= 800 and value < 1000:
        return 5
    elif value >= 710 and value < 740:
        return 6
    elif value >= 580 and value < 630 or np.floor(value) == 788:
        return 7
    elif value >= 140 and value < 240:
        return 8
    else:
        return 0
