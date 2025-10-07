import open3d as o3d
from util import rsrcpath
import numpy as np

modelname = "modelnet38_train_1024pts_fps.dat"
labelname = "modelnet38_shape_names.txt"

modelpath = rsrcpath + "/model/Packages38/" + modelname
labelpath = rsrcpath + "/model/Packages38/" + labelname

datapoint = np.load(modelpath, allow_pickle=True)
datalabel = np.loadtxt(labelpath, dtype=str)

modelpoints = datapoint[0]
modelid = datapoint[1]

id = 10
p1 = modelpoints[id]
idname = datalabel[modelid[id]]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(p1[:, 0:3])
pcd.colors = o3d.utility.Vector3dVector(p1[:, 3:6])
o3d.visualization.draw(pcd, title=str(idname))
