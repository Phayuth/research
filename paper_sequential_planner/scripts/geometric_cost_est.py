import numpy as np
from geometric_ellipse import *

np.random.seed(42)



q1 = np.array([-1.0, 2.5])
q2 = np.array([1.0, 2.5])

q3 = np.array([0.15, 0.60])
q4 = np.array([2.5, 1.5])

q5 = np.array([-2.5, -1.5])
q6 = np.array([2.40, -0.4])

q7 = np.array([-2.0, 2.5])
q8 = np.array([1.0, -2.0])

q9 = np.array([-3.0, 0.0])
q10 = np.array([-3.0, 2.5])

limit = [[-np.pi, np.pi], [-np.pi, np.pi]]
limitmax = limit[0][1] - limit[0][0]
qs = np.array([-2.5, -2.5])
qg = np.array([2.5, 2.5])
cmin = np.linalg.norm(qg - qs)


