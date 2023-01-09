import numpy as np

def euler_to_quat(x, y, z):
  sx, cx = np.sin(x / 2), np.cos(x / 2)
  sy, cy = np.sin(y / 2), np.cos(y / 2)
  sz, cz = np.sin(z / 2), np.cos(z / 2)

  qw = cx * cy * cz + sx * sy * sz
  qx = sx * cy * cz - cx * sy * sz
  qy = cx * sy * cz + sx * cy * sz
  qz = cx * cy * sz - sx * sy * cz

  return np.array([qx, qy, qz, qw])
