import argparse
import os
import sys
sys.path.append('/Users/dragos/Licenta/Thesis')
from Projection3DGS.sfm_utils import run_ffmpeg, run_colmap, clear_directory
from Projection3DGS.gs_segmentation import run_projection
import subprocess
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from torchvision.utils import draw_segmentation_masks
import json

def run_(args):
    print(args.video_path)
    root_dir = os.path.dirname(os.path.abspath(args.video_path))

    if args.quality == 'low':
        fps = 3
        steps = 4000
    elif args.quality == 'medium':
        fps = 4
        steps = 6000
    elif args.quality == 'high':
        fps = 5
        steps = 12000

    if args.ffmpeg:
        output_path = os.path.join(root_dir, 'frames')
        print(f"Output path: {output_path}")
        print("Running ffmpeg...")
        run_ffmpeg(args.video_path, output_path, fps)
    else:
        print("No ffmpeg step requested. Skipping...")
    
    
    if args.colmap:
        #Run COLMAP
        print("Running COLMAP...")

        frames_path = os.path.join(root_dir, 'frames')
        run_colmap(frames_path)
    else:
        print("No COLMAP step requested. Skipping...")

    if args.gauss:
        #Run 3DGS
        build_dir = '/Users/dragos/Licenta/OpenSplat/build'
        opensplat_executable = './opensplat'
        
        splat_path = os.path.join(root_dir, 'splats')
        try:
            if not os.path.exists(splat_path):
                os.makedirs(splat_path)
            else:
                clear_directory(splat_path)
        except Exception as e:
            print(f"Error creating directory {splat_path}: {e}")

        splat_args = [root_dir, '-n', str(steps), '-o', os.path.join(splat_path, "3dgs.ply")] 
        print("Running OpenSplat...")
        subprocess.run([opensplat_executable] + splat_args, cwd=build_dir, check=True)
    else:
        print("No 3DGS step requested. Skipping...")

    # hard_coded_idxs = [1]#, 42, 103]
    segmentation_idxs = []
    if args.segment:
        camera_path = os.path.join(root_dir, 'splats', 'cameras.json')
        with open(camera_path, "r") as f:
            cameras = json.load(f)
        img_list = [camera['img_name'] for camera in cameras]

        n = len(img_list)
        k = int(0.2 * n)
        step = n // k 
        img_list = img_list[::step]

        masks_path = os.path.join(root_dir, 'masks')
        pred_path = os.path.join(root_dir, 'predictions')
        try:
            if not os.path.exists(masks_path):
                os.makedirs(masks_path)
            else:
                clear_directory(masks_path)
        except Exception as e:
            print(f"Error creating directory {masks_path}: {e}")

        try:
            if not os.path.exists(pred_path):
                os.makedirs(pred_path)
            else:
                clear_directory(pred_path)
        except Exception as e:
            print(f"Error creating directory {masks_path}: {e}")

        model_path = '/Users/dragos/Licenta/Results/YOLO/best.pt'
        model = YOLO(model_path)
        device = torch.device('mps')
        model.to(device)

        images_path = os.path.join(root_dir, 'images')
        # for sample_idx in hard_coded_idxs:
        for img_name in img_list:
            sample_idx = int(img_name.split('.')[0])
            # image_path = os.path.join(images_path, f'{sample_idx:04d}.jpg')
            image_path = os.path.join(images_path, img_name)
            results = model(image_path)
            result = results[0]

            if result.masks is not None:
                segmentation_idxs.append(sample_idx)
                predictions = result.boxes.cls
                masks = result.masks
                classes = result.names

                pred_name = f'{sample_idx:04d}'
                for pred_idx, pred in enumerate(predictions):
                    mask = masks.data[pred_idx]
                    mask = mask.cpu().numpy()
                    label = classes[pred.item()]
                    pred_name = pred_name + '-' + label
                    mask_path = os.path.join(masks_path, f'{sample_idx:04d}_{label}.png')
                    cv2.imwrite(mask_path, mask.astype(np.uint8))

                result.save(os.path.join(pred_path, f'{sample_idx:04d}.jpg'))

    if args.project:
        args = argparse.Namespace(
            root_path=root_dir,
            idxs=segmentation_idxs
        )
        run_projection(args)


    print("All steps completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to build a 3DGS car model with damage labels.')

    parser.add_argument('-v', '--video_path',
                        required=True,
                        type=str, 
                        help='(Required) path to the scene data to use.')
    parser.add_argument('-q', '--quality',
                        required=True,
                        type=str, 
                        help='(Required) quality of the reconstruction.')
    parser.add_argument('-ffmpeg', action='store_true',
                    help='Run the ffmpeg step')
    parser.add_argument('-colmap', action='store_true',
                    help='Run the COLMAP step')
    parser.add_argument('-gauss', action='store_true',
                    help='Run the 3DGS step')
    parser.add_argument('-segment', action='store_true',
                    help='Run the segmentation step')
    parser.add_argument('-project', action='store_true',
                    help='Run the projection step')

    args = parser.parse_args()

    run_(args)