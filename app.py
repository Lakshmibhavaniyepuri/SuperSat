
from flask import Flask, request, render_template, send_file, redirect, url_for
import rasterio
import numpy as np
import onnxruntime as ort
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

onnx_model = "C:/Users/Bhava/OneDrive/Desktop/tool/super/s2v2x2_spatrad.onnx"
session = ort.InferenceSession(onnx_model)
factor = 2

@app.route("/")
def index():
    return render_template("index.html")
@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/apply", methods=["GET", "POST"])
def apply():
    if request.method == "POST":
        file = request.files["file"]
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)

        # Run super-resolution
        output_filename = f"SR_{filename}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        run_super_resolution(input_path, output_path)

        return render_template("apply.html", filename=output_filename)

    return render_template("apply.html")

@app.route("/download/<filename>")
def download(filename):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    return send_file(file_path, as_attachment=True)

def run_super_resolution(mosaic_tif, output_tif):
    with rasterio.open(mosaic_tif) as src:
        images = [src.read(i+1).astype(np.float32) for i in range(4)]
        profile = src.profile

    input_array = np.stack(images, axis=0)
    input_array = np.expand_dims(input_array, axis=0)

    outputs = session.run(None, {"input": input_array})
    sr_image = outputs[0][0]

    profile.update(
        height=sr_image.shape[1],
        width=sr_image.shape[2],
        transform=rasterio.Affine(profile['transform'].a / factor, 0, profile['transform'].c,
                                  0, profile['transform'].e / factor, profile['transform'].f)
    )

    with rasterio.open(output_tif, 'w', **profile) as dst:
        for i in range(sr_image.shape[0]):
            dst.write(sr_image[i], i+1)

if __name__ == "__main__":
    app.run(debug=True)
