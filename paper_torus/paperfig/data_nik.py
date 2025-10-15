import numpy as np
from spatial_geometry.utils import Utils


limt6 = np.array(
    [
        [-2 * np.pi, 2 * np.pi],
        [-2 * np.pi, 2 * np.pi],
        [-np.pi, np.pi],
        [-2 * np.pi, 2 * np.pi],
        [-2 * np.pi, 2 * np.pi],
        [-2 * np.pi, 2 * np.pi],
    ]
)
qs = np.array([0.0, -2, 2, 2, -2, 1.2])
qenik = np.array([-1, -1, 3, 2, 2, 2])
qalt = Utils.find_alt_config(qenik.reshape(-1, 1), limt6, filterOriginalq=False).T


print("qstart", qs)
print("qenik", qenik)
print("qalt", qalt)


def distance_on_euclidean(qs, qe):
    diff = np.abs(qe - qs)
    return diff


def distance_on_torus(qa, qb):
    qa = qa.reshape(-1, 1)
    qb = qb.reshape(-1, 1)
    L = np.full_like(qa, 2 * np.pi)
    delta = np.abs(qa - qb)
    deltaw = L - delta
    deltat = np.min(np.hstack((delta, deltaw)), axis=1)
    return np.abs(deltat)


difftorus = np.zeros_like(qalt)
diffeul = np.zeros_like(qalt)
for i in range(qalt.shape[0]):
    difftorus[i, :] = distance_on_torus(qs, qalt[i, :])
    diffeul[i, :] = distance_on_euclidean(qs, qalt[i, :])

deul = distance_on_euclidean(qs, qenik)
dtorus = distance_on_torus(qs, qenik)

print("deul", deul)
print("dtorus", dtorus)
print("diffeul", diffeul)
print("difftorus", difftorus)
