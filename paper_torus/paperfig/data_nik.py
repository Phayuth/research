import numpy as np
from spatial_geometry.utils import Utils
import pandas as pd

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
qs = np.array(
    [
        0.39683294296264565,
        -0.9920818805694598,
        1.5873310565948522,
        -0.8928737640380842,
        1.8518865108490026,
        -3.141592741012566,
    ]
)
qenik = np.array(
    [
        -3.6376335620880056,
        -1.7857475280761683,
        -1.7196087837219274,
        -2.1825799942016664,
        -1.9180250167846644,
        -0.2645549774169931,
    ]
)
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


# distance of original value in euclidean
deul = distance_on_euclidean(qs, qenik)
# distance of original value on torus
dtorus = distance_on_torus(qs, qenik)


# distance of alternative values on torus and euclidean
difftorus = np.zeros_like(qalt)
diffeul = np.zeros_like(qalt)
for i in range(qalt.shape[0]):
    difftorus[i, :] = distance_on_torus(qs, qalt[i, :])
    diffeul[i, :] = distance_on_euclidean(qs, qalt[i, :])

print("deul", deul)
print("dtorus", dtorus)
print("diffeul", diffeul)
print("difftorus", difftorus)

totdeul = np.linalg.norm(deul)
totdtorus = np.linalg.norm(dtorus)
totdiffeul = np.linalg.norm(diffeul, axis=1)
totdifftorus = np.linalg.norm(difftorus, axis=1)

print("totdeul", totdeul)
print("totdtorus", totdtorus)
print("totdiffeul", totdiffeul)
print("totdifftorus", totdifftorus)

# on torus, the minimum distance will always be the same as original


# wirte Qalt to csv
df = pd.DataFrame(qalt)
df.to_csv("data_nik_qalt.csv", index=False)

# write totdiffeul and totdifftorus to csv
df_dist = pd.DataFrame({"totdiffeul": totdiffeul, "totdifftorus": totdifftorus})
df_dist.to_csv("data_nik_dist.csv", index=False)
