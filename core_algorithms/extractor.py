import open3d as o3d
import numpy as np

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return pcd_down, pcd_fpfh

def extract_equipment(raw_pcd, standard_pcd, voxel_size=0.05):
    source_down, source_fpfh = preprocess_point_cloud(standard_pcd, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(raw_pcd, voxel_size)
    
    # 1. 全局配准 (RANSAC)
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold=voxel_size * 1.5,
        estimators=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                  o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
    # 2. 局部精确配准 (ICP)
    result_icp = o3d.pipelines.registration.registration_icp(
        source_down, target_down, voxel_size * 0.4, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    # 3. 裁剪
    standard_pcd.transform(result_icp.transformation)
    bbox = standard_pcd.get_axis_aligned_bounding_box()
    bbox.scale(1.1, bbox.get_center()) # 扩大10%防边缘遗漏
    
    extracted_pcd = raw_pcd.crop(bbox)
    return extracted_pcd, standard_pcd