import open3d as o3d
import numpy as np
import os

def load_point_cloud(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")

    ext = os.path.splitext(file_path)[-1].lower()

    if ext in ['.ply', '.pcd', '.xyz', '.pts']:
        pcd = o3d.io.read_point_cloud(file_path)
    elif ext in ['.las', '.laz']:
        import laspy
        las = laspy.read(file_path)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    elif ext == '.txt':
        points = np.loadtxt(file_path, delimiter=None, usecols=(0, 1, 2)) 
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    else:
        raise ValueError(f"不支持的点云格式: {ext}")

    print(f"✅ 成功加载点云: {file_path}，共 {len(pcd.points)} 个点")
    return pcd

def save_point_cloud(pcd, file_path):
    o3d.io.write_point_cloud(file_path, pcd)