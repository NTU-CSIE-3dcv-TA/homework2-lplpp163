from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import random
import cv2
import time
import open3d as o3d
from tqdm import tqdm
import argparse

np.random.seed(1428) # do not change this seed
random.seed(1428) # do not change this seed

def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

def p3p(K, pts_2d, pts_3d):
    """
    Grunert's P3P implementation (robust & K-optional).
    pts_2d: Nx2 (N=3) - can be in pixel or normalized coordinates
    pts_3d: Nx3 (world coordinates)
    K: 3x3 intrinsic matrix, or None if pts_2d are normalized
    Returns list of (R, T) such that X_cam = R @ X_world + T
    """
    if pts_2d.shape[0] != 3 or pts_3d.shape[0] != 3:
        raise ValueError("P3P requires exactly 3 points")

    # ===== Step 1. Compute bearing vectors =====
    j = np.zeros((3, 3))
    if K is not None:
        # pixel -> normalized
        invK = np.linalg.inv(K)
        for i in range(3):
            p_hom = np.append(pts_2d[i], 1)
            j[i] = invK @ p_hom
            j[i] /= np.linalg.norm(j[i])
    else:
        # already normalized coords
        for i in range(3):
            j[i] = np.array([pts_2d[i, 0], pts_2d[i, 1], 1.0])
            j[i] /= np.linalg.norm(j[i])

    # ===== Step 2. Pre-compute geometry =====
    a = np.linalg.norm(pts_3d[1] - pts_3d[2])
    b = np.linalg.norm(pts_3d[0] - pts_3d[2])
    c = np.linalg.norm(pts_3d[0] - pts_3d[1])
    if a < 1e-6 or b < 1e-6 or c < 1e-6:
        return []

    cos_alpha = np.dot(j[1], j[2])
    cos_beta  = np.dot(j[0], j[2])
    cos_gamma = np.dot(j[0], j[1])

    # ===== Step 3. Grunert quartic coefficients =====
    tmp1 = (a**2 - c**2) / b**2
    tmp2 = (a**2 + c**2) / b**2
    tmp3 = (b**2 - c**2) / b**2
    tmp4 = (b**2 - a**2) / b**2

    A4 = (tmp1 - 1)**2 - (4 * c**2 / b**2) * cos_alpha**2
    A3 = 4 * (tmp1 * (1 - tmp1) * cos_beta - (1 - tmp2) * cos_alpha * cos_gamma + 2 * (c**2 / b**2) * cos_alpha**2 * cos_beta)
    A2 = 2 * (tmp1**2 - 1 + 2 * tmp1**2 * cos_beta**2 + 2 * tmp3 * cos_alpha**2 - 4 * tmp2 * cos_alpha * cos_beta * cos_gamma + 2 * tmp4 * cos_gamma**2)
    A1 = 4 * (-tmp1 * (1 + tmp1) * cos_beta + 2 * (a**2 / b**2) * cos_gamma**2 * cos_beta - (1 - tmp2) * cos_alpha * cos_gamma)
    A0 = (1 + tmp1)**2 - 4 * (a**2 / b**2) * cos_gamma**2

    coeffs = [A4, A3, A2, A1, A0]
    roots = np.roots(coeffs)
    real_roots = roots[np.isreal(roots)].real
    valid_vs = [v for v in real_roots if v > 0]

    # ===== Step 4. Compute possible camera poses =====
    poses = []
    for v in valid_vs:
        denom1 = 1 + v**2 - 2 * v * cos_beta
        if denom1 <= 0: continue
        s1_sq = b**2 / denom1
        if s1_sq <= 0: continue

        denom2 = c**2 / s1_sq
        disc = cos_gamma**2 - (1 - denom2)
        if disc < 0: continue
        sqrt_disc = np.sqrt(disc)
        u_candidates = [cos_gamma + sqrt_disc, cos_gamma - sqrt_disc]

        for u in u_candidates:
            if u <= 0: continue
            denom3 = u**2 + v**2 - 2 * u * v * cos_alpha
            if denom3 <= 0: continue
            check_s1_sq = a**2 / denom3
            if abs(check_s1_sq - s1_sq) > 1e-4:
                continue

            s1 = np.sqrt(s1_sq)
            s2 = u * s1
            s3 = v * s1
            P_cam = np.array([s1 * j[0], s2 * j[1], s3 * j[2]])

            # ===== Step 5. Procrustes alignment =====
            c_cam   = np.mean(P_cam, axis=0)
            c_world = np.mean(pts_3d, axis=0)
            H = (P_cam - c_cam).T @ (pts_3d - c_world)
            U, _, Vt = np.linalg.svd(H)
            R_est = U @ Vt
            if np.linalg.det(R_est) < 0:
                Vt[2, :] *= -1
                R_est = U @ Vt
            T_est = c_cam - R_est @ c_world
            poses.append((R_est, T_est))

    return poses


def undistort_points(pts, K, distCoeffs, max_iter=5):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    k1, k2, p1, p2 = distCoeffs[:4]
    k3 = distCoeffs[4] if len(distCoeffs) > 4 else 0.0

    undistorted = []
    for u, v in pts:
        # Step 1: normalize
        x = (u - cx) / fx
        y = (v - cy) / fy

        # Step 2: iterative undistortion
        for _ in range(max_iter):
            r2 = x*x + y*y
            radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
            x_tang = 2*p1*x*y + p2*(r2 + 2*x*x)
            y_tang = p1*(r2 + 2*y*y) + 2*p2*x*y
            x_est = (x - x_tang) / radial
            y_est = (y - y_tang) / radial
            x, y = x_est, y_est

        undistorted.append([x, y])
    return np.array(undistorted)

def project_points(pts_3d, R, T, K):
    cam_pts = (R @ pts_3d.T + T[:, None]).T
    mask = cam_pts[:, 2] > 1e-6
    proj = np.zeros((len(pts_3d), 2))
    proj[mask] = (K @ cam_pts[mask].T)[:2, :].T / cam_pts[mask, 2][:, None]
    proj[~mask] = np.nan
    return proj

def ransac_p3p(K, pts_2d, pts_3d, distCoeffs=None,
               reproj_thresh=2.0, max_iters=2000,
               confidence=0.99, refine_iters=10):

    n = len(pts_2d)
    best_inliers = 0
    best_pose = None

    # === (1) 預先計算 undistorted normalized 座標，用於 P3P (不乘 K) ===
    if distCoeffs is not None:
        pts_2d_norm = undistort_points(pts_2d, K, distCoeffs)  # normalized (x/z, y/z)
    else:
        pts_2d_norm = (np.linalg.inv(K) @ np.hstack([pts_2d, np.ones((n, 1))]).T).T[:, :2]

    for i in range(max_iters):
        idx = np.random.choice(n, 4, replace=False)
        idx_p3p, idx_val = idx[:3], idx[3]

        # --- P3P 使用 normalized coord (不乘 K) ---
        candidates = p3p(None, pts_2d_norm[idx_p3p], pts_3d[idx_p3p])  # p3p不再用K

        for R_cand, T_cand in candidates:
            # 正向性
            if np.mean((R_cand @ pts_3d.T + T_cand[:, None])[2]) < 0:
                continue

            # 驗證第4點 (project_points回像素空間)
            proj_val = project_points(pts_3d[[idx_val]], R_cand, T_cand, K)
            err_val = np.linalg.norm(proj_val - pts_2d[[idx_val]])
            if err_val > reproj_thresh * 2:
                continue

            # 計算全部誤差
            proj = project_points(pts_3d, R_cand, T_cand, K)
            errors = np.linalg.norm(proj - pts_2d, axis=1)
            inliers = np.where(errors < reproj_thresh)[0]
            if len(inliers) > best_inliers:
                best_inliers = len(inliers)
                best_pose = (R_cand, T_cand, inliers)

        if best_inliers / n > confidence:
            break

    if best_pose is None:
        return None, None, 0

    R_best, T_best, inliers = best_pose
    rvec = R.from_matrix(R_best).as_rotvec()
    tvec = T_best.copy()

    return rvec, tvec, best_inliers


def pnpsolver(query,model,cameraMatrix=0,distortion=0,use_opencv=False):
    kp_query, desc_query = query
    kp_model, desc_model = model
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])

    # TODO: solve PnP problem using Opencv2
    # Hint: you may use "Descriptors Matching and ratio test" first
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(desc_query, desc_model, k=2)
    good_matches = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 6:
        return False, None, None, None

    # Extract 2D-3D correspondences
    pts_2d = np.float32([kp_query[m.queryIdx] for m in good_matches])
    pts_3d = np.float32([kp_model[m.trainIdx] for m in good_matches])

    # --------------------------
    # 2. Solve PnP
    # --------------------------
    if use_opencv:
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, cameraMatrix, distCoeffs,
            iterationsCount=2000,
            reprojectionError=2.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_P3P
        )
        if not retval:
            return False, None, None, None
        return True, rvec.flatten(), tvec.flatten(), len(inliers)
    else:
        rvec, tvec, inliers = ransac_p3p(cameraMatrix, pts_2d, pts_3d, distCoeffs)
        if rvec is None:
            return False, None, None, None
        return True, rvec, tvec, inliers

def rotation_error(R1, R2):
    #TODO: calculate rotation error
    R1 = R.from_quat(R1.flatten()).as_matrix()
    R2 = R.from_quat(R2.flatten()).as_matrix()
    rel_R = R1 @ R2.T
    trace = np.trace(rel_R)
    angle = np.arccos(np.clip((trace - 1)/2, -1, 1)) * 180 / np.pi
    return angle

def translation_error(t1, t2):
    #TODO: calculate translation error
    return np.linalg.norm(t1 - t2)

def visualization(Camera2World_Transform_Matrixs, points3D_df):

    # ---------------------------
    # 1. 準備點雲
    # ---------------------------
    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB']) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # ---------------------------
    # 2. 相機金字塔 (frustum)
    # ---------------------------
    line_sets = []
    # frustum 大小
    frustum_scale = 0.1
    for c2w in Camera2World_Transform_Matrixs:
        frustum_verts = np.array([
            [0,0,0],
            [-0.5,-0.5,1], [0.5,-0.5,1],
            [0.5,0.5,1], [-0.5,0.5,1]
        ]) * frustum_scale

        frustum_world = (c2w[:3,:3] @ frustum_verts.T).T + c2w[:3,3]
        lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(frustum_world),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.paint_uniform_color([1,0,0])  # 紅色相機
        line_sets.append(line_set)

    # ---------------------------
    # 3. 相機軌跡
    # ---------------------------
    traj_points = [c2w[:3,3] for c2w in Camera2World_Transform_Matrixs]
    traj_lines = [[i, i+1] for i in range(len(traj_points)-1)]
    traj_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(traj_points),
        lines=o3d.utility.Vector2iVector(traj_lines)
    )
    traj_set.paint_uniform_color([0,1,0])  # 綠色軌跡

    # ---------------------------
    # 4. 顯示
    # ---------------------------
    o3d.visualization.draw_geometries([pcd] + line_sets + [traj_set],
                                      zoom=0.7,
                                      front=[0,0,-1],
                                      lookat=[0,0,0],
                                      up=[0,-1,0])

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--q1", action="store_true", help="Run Q1: estimate poses")
    parser.add_argument("--q2", action="store_true", help="Run Q2: AR visualization")
    parser.add_argument("--vis", action="store_true", help="Run Open3D visualization")
    args = parser.parse_args()

    if args.q1:
        # Load datas
        images_df = pd.read_pickle("data/images.pkl")
        train_df = pd.read_pickle("data/train.pkl")
        points3D_df = pd.read_pickle("data/points3D.pkl")
        point_desc_df = pd.read_pickle("data/point_desc.pkl")

        # Process model descriptors
        desc_df = average_desc(train_df, points3D_df)
        kp_model = np.array(desc_df["XYZ"].to_list())
        desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)

        IMAGE_ID_LIST = (
            images_df[images_df["NAME"].str.contains("valid")]
            .assign(NUM=lambda df: df["NAME"].str.extract(r'(\d+)').astype(int))
            .sort_values("NUM")["IMAGE_ID"]
            .tolist()
        )

        # IMAGE_ID_LIST = [200,201]
        r_list = []
        t_list = []
        rotation_error_list = []
        translation_error_list = []
        est_poses = {}
        
        for idx in tqdm(IMAGE_ID_LIST):
            # Load quaery image
            fname = (images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values[0]
            rimg = cv2.imread("data/frames/" + fname, cv2.IMREAD_GRAYSCALE)

            # Load query keypoints and descriptors
            points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == idx]
            kp_query = np.array(points["XY"].to_list())
            desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

            # Find correspondance and solve pnp
            retval, rvec, tvec, inliers = pnpsolver((kp_query, desc_query), (kp_model, desc_model), use_opencv=False)
            
            if not retval:
                print(f"Warning: PnP failed for IMAGE_ID={idx}")
                continue  # 跳過這張影像
            
            # rotq = R.from_rotvec(rvec.reshape(1,3)).as_quat() # Convert rotation vector to quaternion
            # tvec = tvec.reshape(1,3) # Reshape translation vector
            r_list.append(rvec)
            t_list.append(tvec)
            est_poses[idx] = (rvec, tvec)

            # Get camera pose groudtruth
            ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
            rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
            tvec_gt = ground_truth[["TX","TY","TZ"]].values

            # Calculate error
            r_est_quat = R.from_rotvec(rvec).as_quat()
            r_error = rotation_error(rotq_gt, r_est_quat)
            t_error = translation_error(tvec_gt, tvec)
            
            rotation_error_list.append(r_error)
            translation_error_list.append(t_error)

        # TODO: calculate median of relative rotation angle differences and translation differences and print them
        if rotation_error_list:
            median_rot = np.median(rotation_error_list)
            median_trans = np.median(translation_error_list)
            print(f"Median Rotation Error: {median_rot:.3f} degrees")
            print(f"Median Translation Error: {median_trans:.3f} units")

        # TODO: result visualization
        Camera2World_Transform_Matrixs = []
        for r, t in zip(r_list, t_list):
            # TODO: calculate camera pose in world coordinate system
            R_cam = R.from_rotvec(r).as_matrix()  # W2C rotation
            T_cam = t  # W2C translation
            R_world = R_cam.T
            T_world = -R_world @ T_cam
            c2w = np.eye(4)
            c2w[:3,:3] = R_world
            c2w[:3,3] = T_world
            Camera2World_Transform_Matrixs.append(c2w)
        visualization(Camera2World_Transform_Matrixs, points3D_df)
        
        # Save est_poses for Q2-2
        np.save('est_poses.npy', est_poses)
    
    if args.q2:
        # Load est_poses.npy (from Q1)
        est_poses = np.load('est_poses.npy', allow_pickle=True).item()  # Dict: {IMAGE_ID: (rvec, tvec)}

        # Load images_df and filter test set (sorted)
        images_df = pd.read_pickle("data/images.pkl")
        test_df = (
            images_df[images_df["IMAGE_ID"].isin(est_poses.keys())]
            .assign(NUM=lambda df: df["NAME"].str.extract(r'(\d+)').astype(int))
            .sort_values("NUM")
            .drop(columns="NUM")
        )

        # Camera params
        K = np.array([[1868.27, 0, 540], [0, 1869.18, 960], [0, 0, 1]])
        distCoeffs = np.array([0.0847023, -0.192929, -0.000201144, -0.000725352])

        # Load cube vertices (from transform_cube.py)
        cube_vertices_world = np.load('cube_vertices.npy')  # (8,3)

        # 定義 cube 的 6 個面 (每個面由 4 個頂點 index 組成)
        cube_faces = [
            [0, 1, 3, 2],  # bottom (z=0)
            [4, 5, 7, 6],  # top    (z=1)
            [0, 1, 5, 4],  # front  (y=0)
            [2, 3, 7, 6],  # back   (y=1)
            [0, 2, 6, 4],  # left   (x=0)
            [1, 3, 7, 5]   # right  (x=1)
        ]
        
        # 每面一個顏色 (BGR, OpenCV 用的順序)
        face_colors = [
            (0,   0, 255),   # red
            (0, 255,   0),   # green
            (255, 0,   0),   # blue
            (0, 255, 255),   # yellow
            (255, 0, 255),   # magenta
            (255, 255, 0)    # cyan
        ]
        
        # Video writer (寬1080 高1920)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('ar_video_est.mp4', fourcc, 10.0, (1080, 1920))  # (w,h)
        h, w = 1920, 1080

        print(f"Image size: w={w}, h={h}")

        # --- 幫助函式: 在四邊形面上均勻取樣點 ---
        def sample_face_points(v0, v1, v2, v3, density=20):
            pts = []
            for i in range(density + 1):
                for j in range(density + 1):
                    u = i / density
                    v = j / density
                    # 雙線性插值
                    p = (1-u)*(1-v)*v0 + u*(1-v)*v1 + (1-u)*v*v2 + u*v*v3
                    pts.append(p)
            return np.array(pts, dtype=np.float32)

        for frame_idx, row in enumerate(test_df.iterrows()):
            _, row = row
            idx = row["IMAGE_ID"]
            fname = row["NAME"]
            img = cv2.imread(f"data/frames/{fname}")
            if img is None:
                print(f"Warning: Skipped {fname}")
                continue

            rvec, tvec = est_poses[idx]

            rvec = np.asarray(rvec, dtype=np.float64)
            tvec = np.asarray(tvec, dtype=np.float64)

            # cube 在相機座標系下
            R_w2c = R.from_rotvec(rvec).as_matrix()

            # 每個面打點 & 投影
            for face_idx, face in enumerate(cube_faces):
                v_raw = [cube_vertices_world[i] for i in face]
                
                # 依照左上, 右上, 左下, 右下排列
                v0 = v_raw[0]  # 左上
                v1 = v_raw[1]  # 右上
                v2 = v_raw[3]  # 左下
                v3 = v_raw[2]  # 右下
                
                face_points = sample_face_points(v0, v1, v2, v3, density=15)

                # 投影
                pts_2d, _ = cv2.projectPoints(face_points, rvec, tvec, K, distCoeffs)
                pts_2d = pts_2d.squeeze()
                
                # 過濾影像外的點
                mask = (pts_2d[:,0] >= 0) & (pts_2d[:,0] < w) & \
                    (pts_2d[:,1] >= 0) & (pts_2d[:,1] < h)
                    
                # 過濾fov外的點
                face_points_cam = (R_w2c @ face_points.T + tvec.reshape(3,1)).T                  
                fov_x = np.arctan2(face_points_cam[:,0], face_points_cam[:,2])
                fov_y = np.arctan2(face_points_cam[:,1], face_points_cam[:,2])
                max_angle_x = np.arctan2(w/2, K[0,0])
                max_angle_y = np.arctan2(h/2, K[1,1])
                mask_fov = (np.abs(fov_x) <= max_angle_x) & (np.abs(fov_y) <= max_angle_y)
                mask = mask & mask_fov

                      
                pts_2d = pts_2d[mask].astype(int)
                
                # 畫點
                color = face_colors[face_idx]
                for (x,y) in pts_2d:
                    cv2.circle(img, (x,y), 10, color, -1)
                    
            out.write(img)

        out.release()
        print("AR video saved as 'ar_video_est.mp4'")

    # 只在 Q1/Q2 執行完後才進行視覺化
    if args.vis:
        est_poses = np.load('est_poses.npy', allow_pickle=True).item()
        points3D_df = pd.read_pickle("data/points3D.pkl")

        Camera2World_Transform_Matrixs = []
        for rvec, tvec in est_poses.values():
            R_cam = R.from_rotvec(rvec).as_matrix()
            T_cam = tvec
            R_world = R_cam.T
            T_world = -R_world @ T_cam
            c2w = np.eye(4)
            c2w[:3,:3] = R_world
            c2w[:3,3] = T_world
            Camera2World_Transform_Matrixs.append(c2w)

        visualization(Camera2World_Transform_Matrixs, points3D_df)