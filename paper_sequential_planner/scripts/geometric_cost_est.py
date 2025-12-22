import numpy as np
import matplotlib.pyplot as plt
from geometric_ellipse import *


np.random.seed(42)

limit = [[-np.pi, np.pi], [-np.pi, np.pi]]
limitmax = limit[0][1] - limit[0][0]
qs = np.array([-2.5, -2.5])
qg = np.array([2.5, 2.5])
cmin = np.linalg.norm(qg - qs)
