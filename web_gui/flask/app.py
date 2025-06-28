from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
import os
from segmentation.run_script import run_
import argparse
import json
import time
from pathlib import Path

upload_root = Path(__file__).resolve().parents[1] / 'demo'
new_sample_root = os.path.join(upload_root, 'new_sample')

app = Flask(__name__)
CORS(app)

# upload_root = '/Users/dragos/Licenta/data/presentation_demo'
# new_sample_root = '/Users/dragos/Licenta/data/presentation_demo/new_sample'


@app.route('/get-sample-wm/<path:sample_dir>/<filename>')
def get_sample_wm(sample_dir, filename):
    pred_path = os.path.join(upload_root, sample_dir, 'predictions')
    response = make_response(send_from_directory(pred_path, filename))
    response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

@app.route('/get-sample-wom/<path:sample_dir>/<filename>')
def get_sample_wom(sample_dir, filename):
    pred_path = os.path.join(upload_root, sample_dir, 'images')
    response = make_response(send_from_directory(pred_path, filename))
    response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

@app.route('/get-samples-data/<path:sample_dir>')
def image_data(sample_dir):
    files = []
    camera_path = os.path.join(upload_root, sample_dir, 'splats', 'cameras.json')
    with open(camera_path, "r") as f:
        cameras = json.load(f)


    for f in os.listdir(os.path.join(upload_root, sample_dir, 'predictions')):
        print('Searching for predictions in:', os.path.join(upload_root, sample_dir, 'predictions'))
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            print('Image file is: ', f)
            title_idx = os.path.splitext(f)[0]

            # title_split = title.split('-', 1)
            # title_idx = title_split[0]
            # parts = title_split[1]

            camera = [camera for camera in cameras if camera['img_name'] == f][0]

            pos_list = camera['position']

            camera_pos = ','.join(f"{x:.4f}" for x in pos_list)

            # labels = parts.split('-')
            # labels = [label.replace('_', ' ').title() for label in labels]
            # caption = ', '.join(labels)

            files.append(
                {
                    "filename": f,
                    "url": f"http://localhost:5100/get-sample-wm/{sample_dir}/{f}",
                    "camera": camera_pos,
                }
            )

    return jsonify(files)

@app.route('/splats/<path:sample_dir>/<filename>')
def serve_splat(sample_dir, filename):
    return send_from_directory(os.path.join(upload_root, sample_dir, 'splats'), filename)

@app.route('/delete-pred/<path:sample_dir>/<filename>', methods=['DELETE'])
def delete_splat(sample_dir, filename):
    pred_name = filename + '.jpg'
    file_path = os.path.join(upload_root, sample_dir, 'predictions', pred_name)
    try:
        os.remove(file_path)
        return jsonify({"status": "success", "message": f"Deleted {filename}"}), 200
    except FileNotFoundError:
        return jsonify({"status": "error", "message": "File not found"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():

    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']
    quality = request.form['quality']
    print("Desired quality: ", quality)
    
    if file.filename == '':
        return jsonify({'message': 'No file selected'}), 400
    os.makedirs(new_sample_root, exist_ok=True)
    filepath = os.path.join(new_sample_root, file.filename)
    file.save(filepath)

    args = argparse.Namespace(
        video_path=filepath,
        quality=quality,
        ffmpeg=True,
        colmap=True,
        gauss=True,
        segment=True,
        project=True,
    )
    start = time.time()
    run_(args)
    elapsed = time.time() - start

    print(f"Function finished in {elapsed:.4f} seconds.")
    
    return jsonify({'message': f'Your video has been processed. Click "Render" to visualize the results.'})

if __name__ == '__main__':
    app.run(debug=True, port=5100)
