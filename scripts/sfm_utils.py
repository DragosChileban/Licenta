import os
import shutil
import cv2
import math

def clear_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path) 
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def run_ffmpeg(video_path, output_path, fps):
    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        else:
            clear_directory(output_path)
    except Exception as e:
        print(f"Error creating directory {output_path}: {e}")

    ffmpeg_cmd = f"ffmpeg -i {video_path} -qscale:v 1 -qmin 1 -vf fps={str(fps)} {output_path}/%04d.jpg"

    os.system(ffmpeg_cmd)

def run_colmap(frames_path):
    # Resize files
    images = sorted(os.listdir(frames_path))
    image = cv2.imread(os.path.join(frames_path, images[0]))
    height, width = image.shape[:2]
    
    max_side = 640

    if width > height:
        scale = max_side / width
    else:
        scale = max_side / height

    scaled_width = int(width * scale)
    scaled_height = int(height * scale)

    new_width = math.ceil(scaled_width / 32) * 32
    new_height = math.ceil(scaled_height / 32) * 32

    new_size = (new_width, new_height)

    resized_path = os.path.join(os.path.dirname(frames_path), 'images')
    print(f"Resized path: {resized_path}")

    try:
        if not os.path.exists(resized_path):
            os.makedirs(resized_path)
        else:
            clear_directory(resized_path)
    except Exception as e:
        print(f"Error creating directory {resized_path}: {e}")

    for image_name in images:
        image_path = os.path.join(frames_path, image_name)
        image = cv2.imread(image_path)
        if image is not None:
            resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(resized_path, image_name), resized)

    workspace = os.path.dirname(resized_path)
    images_folder = os.path.join(workspace, "images")
    db_path = os.path.join(workspace, "database.db")
    sparse_path = os.path.join(workspace, "sparse")

    try:
        if not os.path.exists(sparse_path):
            os.makedirs(sparse_path)
        else:
            clear_directory(sparse_path)
    except Exception as e:
        print(f"Error creating directory {sparse_path}: {e}")

    
    os.system(f"colmap database_creator --database_path {db_path}")

    os.system(f"colmap feature_extractor --database_path {db_path} --image_path {images_folder}")

    os.system(f"colmap exhaustive_matcher --database_path {db_path}")

    os.system(f"colmap mapper --database_path {db_path} --image_path {images_folder} --output_path {sparse_path}")


