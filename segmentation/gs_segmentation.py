import os
import argparse
import numpy as np
import json
import cv2
from plyfile import PlyData, PlyElement
import time
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def load_ply(file_path, cameras, shrink=True):
    ply_data = PlyData.read(file_path)
    vertex_data = ply_data['vertex']

    minx, maxx = 1e9, -1e9
    miny, maxy = 1e9, -1e9
    minz, maxz = 1e9, -1e9
    for camera in cameras:
        camera_coords = camera['position']
        x, y, z = camera_coords[0], camera_coords[1], camera_coords[2]
        minx = min(minx, x)
        maxx = max(maxx, x)
        miny = min(miny, y)
        maxy = max(maxy, y)
        minz = min(minz, z)
        maxz = max(maxz, z)
    
    coords = np.array([[minx, maxx], [miny, maxy], [minz, maxz]])
    diff = np.array([maxx - minx, maxy - miny, maxz - minz])
    sorted_diff = np.argsort(diff)

    coords[sorted_diff[0]] += coords[sorted_diff[1]]

    mask = (
        (vertex_data['x'] >= coords[0, 0]) & (vertex_data['x'] <= coords[0, 1]) &
        (vertex_data['y'] >= coords[1, 0]) & (vertex_data['y'] <= coords[1, 1]) &
        (vertex_data['z'] >= coords[2, 0]) & (vertex_data['z'] <= coords[2, 1])
    )

    filtered_vertex_data = vertex_data[mask]
    if shrink:
        vertex_data = filtered_vertex_data

    positions = np.vstack([
        vertex_data['x'],
        vertex_data['y'],
        vertex_data['z']
    ]).T
    
    colors = np.vstack([
        vertex_data['f_dc_0'],
        vertex_data['f_dc_1'],
        vertex_data['f_dc_2']
    ]).T

    
    scales = np.vstack([
        vertex_data['scale_0'],
        vertex_data['scale_1'],
        vertex_data['scale_2']
    ]).T
    
    rotations = np.vstack([
        vertex_data['rot_0'],
        vertex_data['rot_1'],
        vertex_data['rot_2'],
        vertex_data['rot_3']
    ]).T
    
    opacities = vertex_data['opacity']
    
    return vertex_data, positions, colors, scales, rotations, opacities

def save_ply(vertex, new_colors, new_opacities, path, new_indexes=None):
    new_vertex_data = []

    if new_indexes is not None:
        vertex = [vertex[i] for i in new_indexes]
        new_colors = new_colors[new_indexes]
        new_opacities = new_opacities[new_indexes]

    for i in range(len(vertex)):
        v = vertex[i]
        new_vertex = (
            v['x'], v['y'], v['z'],
            v['nx'], v['ny'], v['nz'],
            new_colors[i, 0], new_colors[i, 1], new_colors[i, 2],
            *[v[f"f_rest_{j}"] for j in range(45)],
            new_opacities[i],
            v['scale_0'], v['scale_1'], v['scale_2'],
            v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3'],
        )
        new_vertex_data.append(new_vertex)

    # Define the data structure
    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
    ]

    # Add all f_rest_* fields
    vertex_dtype += [(f"f_rest_{i}", 'f4') for i in range(45)]

    # Remaining fields
    vertex_dtype += [
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
    ]

    # Create numpy array
    vertex_array = np.array(new_vertex_data, dtype=vertex_dtype)

    # Create PlyElement and PlyData
    ply_el = PlyElement.describe(vertex_array, 'vertex')
    ply_data = PlyData([ply_el], text=False)

    # Save to disk
    ply_data.write(path)

def get_masks(masks_path, img_name):
    masks = []

    # mask_pallete = {
    #     'dent': [255,192,203], 
    #     'scratch': [2,191,255],
    #     'crack': [25,248,3],
    #     'glass_shatter': [128,29,128], 
    #     'lamp_broken': [255,215,0], 
    #     'tire_flat': [255,4,0]
    # }
    mask_palette = {
        'dent': [2, 1, 2], #good color
        'scratch': [0, 0, 5], #purple not blue
        'crack': [0, 5, 0],
        'glass_shatter': [5, 0, 5],
        'lamp_broken': [5, 5, 0], #good color
        'tire_flat': [5, 0, 0] #good color
    }


    img_idx = os.path.splitext(img_name)[0]
    
    for f in os.listdir(masks_path):
        print('filename is ', f, ' and image idx is ', img_idx)
        if f.lower().startswith(img_idx):
            print(f, 'starts with', img_idx)
            mask_title = os.path.splitext(f)[0]
            mask_label = mask_title.split('_', 1)[1]

            masks.append(
                {
                    'path': f,
                    'color': mask_palette[mask_label],
                }
            )
    return masks


def project_points(points, camera):
    R = camera['rotation']
    t = camera['position']
    
    points_cam = np.dot(points - t, R)
    
    fx, fy = camera['fx'], camera['fy']
    px = fx * points_cam[:, 0] / points_cam[:, 2] + camera['width'] / 2
    py = fy * points_cam[:, 1] / points_cam[:, 2] + camera['height'] / 2
    
    valid = points_cam[:, 2] > 0
    z = points_cam[:, 2]
    
    valid = valid & (px >= 0) & (px < camera['width']) & (py >= 0) & (py < camera['height'])# & (z < 0.4) #the condition for depth
    
    return px, py, valid, z

def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix."""
    w, x, y, z = q
    rotation_matrix = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    return rotation_matrix

def project_only(fx, fy, points_cam, width, height, mask):
    px = fx * points_cam[:, 0] / points_cam[:, 2] + width / 2
    py = fy * points_cam[:, 1] / points_cam[:, 2] + height / 2
    
    valid = points_cam[:, 2] > 0
    
    valid = valid & (px >= 0) & (px < width) & (py >= 0) & (py < height)

    px_valid = np.round(px[valid]).astype(int)
    py_valid = np.round(py[valid]).astype(int)
    
    in_mask = mask[py_valid, px_valid]
    
    mask_indices = np.where(valid)[0][in_mask]
    
    return px, py, points_cam[:, 2], mask_indices


def project_mask(px, py, fx, fy, R, points_cam, scales, rotations, opacities, sorted_indices, width, height, mask_idxs):
    start = time.time()

    shape_time = 0
    collect_time = 0
    check_time = 0

    # mask_sorted_indices = sorted_indices[np.isin(sorted_indices, mask_indices)]
    mask_sorted_indices = sorted_indices[
        np.isin(sorted_indices, mask_idxs)
    ]

    gauss_buffer = np.zeros((height, width), dtype=np.float32)


    z_buffer = {}
    weights_map = {}
    depth_vals = []
    opacity_vals = [1e6]
    op_flag = False

    for idx, gauss_idx in enumerate(mask_sorted_indices):
            check1 = time.time()
            x_proj = px[gauss_idx]
            y_proj = py[gauss_idx]

            # print("Opacity vector ", opacity_vals[0])

            # print("Gauss idx", gauss_idx)

            # print("Checking buffer for vals ", x_proj, y_proj)


            if gauss_buffer[int(y_proj), int(x_proj)] < np.mean(opacity_vals):# - np.std(opacity_vals):

                check2 = time.time() - check1
                check_time += check2
                shape1 = time.time()

                p_cam = points_cam[gauss_idx]
                scale = scales[gauss_idx]
                quat = rotations[gauss_idx]
                opacity = opacities[gauss_idx]

                # print("Opacity", opacity)

                rot_matrix = quaternion_to_rotation_matrix(quat)
                scale_matrix = np.diag(np.exp(scale))
                cov_world = rot_matrix @ scale_matrix @ scale_matrix @ rot_matrix.T
                cov_camera = R @ cov_world @ R.T
                
                z_inv = 1.0 / p_cam[2]
                J = np.array([
                    [fx * z_inv, 0, -fx * p_cam[0] * z_inv * z_inv],
                    [0, fy * z_inv, -fy * p_cam[1] * z_inv * z_inv]
                ])
                cov_image = J @ cov_camera[0:3, 0:3] @ J.T

                eigvals, eigvecs = np.linalg.eigh(cov_image)
                eigvals = np.maximum(eigvals, 1e-6)  # Ensure positive values
                
                # Calculate bounding box
                # Use 3 standard deviations for coverage
                sigma_factor = 3.0
                bbox_radius = np.ceil(sigma_factor * np.sqrt(np.max(eigvals))).astype(int)

                x_min = max(0, int(x_proj - bbox_radius))
                x_max = min(width - 1, int(x_proj + bbox_radius))
                y_min = max(0, int(y_proj - bbox_radius))
                y_max = min(height - 1, int(y_proj + bbox_radius))

                roi_x, roi_y = np.meshgrid(
                np.arange(x_min, x_max + 1),
                np.arange(y_min, y_max + 1),
                indexing='ij'
                )
                roi_x = roi_x.flatten()
                roi_y = roi_y.flatten()
                
                # Calculate Gaussian PDF
                dx = roi_x - x_proj
                dy = roi_y - y_proj
                points = np.vstack([dx, dy]).T
                
                # Calculate Mahalanobis distance
                inv_cov = np.linalg.inv(cov_image)
                mahalanobis_dist = np.sum(points @ inv_cov * points, axis=1)
                
                # Calculate Gaussian weights
                weights = opacity * np.exp(-0.5 * mahalanobis_dist)

                threshold = min(0.05 * opacity, 1e-5)#mahalanobis_dist <= 6#0.01 *opacity  # or 0.1 *opacity
                important_indices = weights > threshold

                important_roi_x = np.array(roi_x[important_indices])
                important_roi_y = np.array(roi_y[important_indices])
                important_weights = np.array(weights[important_indices])

                weights_map[gauss_idx] = (important_weights, important_roi_x, important_roi_y)

                if opacity > 0:
                    opacity_vals.append(opacity)
                    if len(opacity_vals) == 2 and not op_flag:
                        op_flag = True
                        opacity_vals = opacity_vals[1:]

                shape2 = time.time() - shape1

                shape_time += shape2

                collect1 = time.time()

                z_buffer[(int(x_proj), int(y_proj))] = (p_cam[2], gauss_idx, opacity)
                depth_vals.append(p_cam[2])

                np.add.at(gauss_buffer, (important_roi_y, important_roi_x), important_weights)

                collect2 = time.time() - collect1

                collect_time += collect2


    mean_depth = np.mean(depth_vals)
    depth_std = np.std(depth_vals)

    mean_op = np.mean(opacity_vals)
    op_std = np.std(opacity_vals)

    lower_bound = mean_depth - 2 * depth_std
    upper_bound = mean_depth + 1 * depth_std

    op_lower_bound = mean_op - 2 * op_std
    op_upper_bound = mean_op + 2 * op_std

    segmentation_idxs = []

    for (x, y), (z, idx, op) in z_buffer.items():
            if lower_bound <= z <= upper_bound and op_lower_bound <= op <= op_upper_bound:
                segmentation_idxs.append(idx)

    elapsed = time.time() - start

    # print('Function finished in: ', elapsed)


    return segmentation_idxs, z_buffer, weights_map, opacity_vals, elapsed


def run_projection(args):
    root_path = args.root_path
    sample_idxs = args.idxs

    camera_path = os.path.join(root_path, 'splats', 'cameras.json')
    ply_path = os.path.join(root_path, 'splats', '3dgs.ply')
    all_ply_path = os.path.join(root_path, 'splats', 'proj_0000.ply')
    masks_path = os.path.join(root_path, 'masks')

    with open(camera_path, "r") as f:
        cameras = json.load(f)

    vertex_data, positions, colors, scales, rotations, opacities = load_ply(ply_path, cameras)


    all_colors = np.copy(colors)
    all_opacities = np.copy(opacities)

    
    for sample_idx in sample_idxs:
        sample_projection = {}
        proj_time = 0

        new_ply_path = os.path.join(root_path, 'splats', f'proj_{sample_idx:04d}.ply')
        print("Processing sample:", sample_idx)
        camera = [camera for camera in cameras if camera['img_name'] == str(sample_idx).zfill(4) + '.jpg'][0]
        print("Camera was: ", camera['img_name'])

        R = np.array(camera['rotation'])
        t = np.array(camera['position'])

        distances = np.linalg.norm(positions - t.reshape(1, 3), axis=1)
        sorted_indices = np.argsort(distances)
        points_cam = np.dot(positions - t, R)
        fx, fy = camera['fx'], camera['fy']

        masks = get_masks(masks_path, camera['img_name'])

        new_colors = np.copy(colors)
        new_opacities = np.copy(opacities)
        # mask=mask[0]
        
        for mask_idx, mask_dict in enumerate(masks):
            print("Processing mask:", sample_idx)
            mask_path = os.path.join(masks_path, mask_dict['path'])
            try:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                # mask_height, mask_width = mask.shape[:2]
                if mask is None:
                    print(f"[!] Failed to load mask: {mask_path}")
                    continue
            except Exception as e:
                print(f"Error processing mask for {camera['img_name']}: {e}")

            mask = mask > 0

            start = time.time()
            px_, py_, pz_, mask_proj = project_only(fx, fy, points_cam, camera['width'], camera['height'], mask)
            elapsed = time.time() - start
            proj_time += elapsed


            sample_projection[mask_idx] = (px_, py_, mask_proj, pz_)

            segmentation_idxs, z_buffer, weights_map, opacity_vals, seg_time = project_mask(px=px_, py=py_, fx=fx, fy=fy, R=R, points_cam=points_cam, scales=scales, rotations=rotations, opacities=opacities,
                                                                 sorted_indices=sorted_indices, width=camera['width'], height=camera['height'], mask_idxs=mask_proj)
            
            for gauss_idx in segmentation_idxs:
                new_colors[gauss_idx] = mask_dict['color']
                new_opacities[gauss_idx] = 100
                all_colors[gauss_idx] = mask_dict['color']
                all_opacities[gauss_idx] = 100

            save_ply(vertex_data, new_colors, new_opacities, new_ply_path)
        
    
    
            print('Segmentation finished in: ', proj_time+seg_time)


    save_ply(vertex_data, all_colors, all_opacities, all_ply_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to project 2D masks to 3DGS.')

    parser.add_argument('-p', '--root_path',
                        required=True,
                        type=str, 
                        help='(Required) path to the root dir of scene data to use.')
    
    parser.add_argument('-idxs', type=list,
                    help='Index of the labeled image to be projected.')

    args = parser.parse_args()
