from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import base64
import io
from ocr import NeuralNetwork  # your neural net class

app = Flask(__name__)
model = NeuralNetwork()
model.load_weights("model_weights.npy")  # Pretrained weights

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["image"].split(",")[1]
    image = Image.open(io.BytesIO(base64.b64decode(data))).convert("L").resize((28, 28))
    image_array = np.array(image).flatten() / 255.0
    result = model.predict(image_array)
    return jsonify({"digit": int(result)})

if __name__ == "__main__":
    app.run(debug=True)
