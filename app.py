"""Flask API that serves predictions from a model stored in S3.
The model is downloaded once at startup, not on every request.
Credentials are never written here: boto3 picks up the EC2
instance's IAM role automatically.
"""
import os
import tempfile
import boto3
import joblib
from botocore.exceptions import ClientError
from flask import Flask, jsonify, request
BUCKET_NAME = os.environ.get("MODEL_BUCKET", "ml-iris-model-inshaal-ajwa")
MODEL_KEY = os.environ.get("MODEL_KEY", "iris_model.pkl")
# tempfile.gettempdir() resolves to /tmp on Linux and the user's
# Temp folder on Windows, so this works in both places.
LOCAL_PATH = os.path.join(tempfile.gettempdir(), "iris_model.pkl")
CLASS_NAMES = ["setosa", "versicolor", "virginica"]
app = Flask(__name__)
model = None
load_error = None
def load_model():
    """Download the model from S3 once, when the process starts."""
    global model, load_error
    try:
        s3 = boto3.client("s3")
        s3.download_file(BUCKET_NAME, MODEL_KEY, LOCAL_PATH)
        model = joblib.load(LOCAL_PATH)
        print("Model loaded from s3://{}/{}".format(BUCKET_NAME, MODEL_KEY))
    except ClientError as e:
        load_error = "S3 error: {}".format(e.response["Error"]["Code"])
        print("FAILED to load model: {}".format(load_error))
    except Exception as e:
        load_error = "{}: {}".format(type(e).__name__, e)
        print("FAILED to load model: {}".format(load_error))
load_model()
@app.route("/health")
def health():
    """Liveness check. Week 9's load balancer will call this."""
    if model is None:
        return jsonify({"status": "unhealthy", "error": load_error}), 503
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "instance_id": os.environ.get("INSTANCE_ID", "unknown"),
    }), 200
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded", "detail": load_error}), 503
    # force=True so the request still parses if the caller forgot
    # the Content-Type: application/json header.
    body = request.get_json(force=True, silent=True)
    if body is None:
        return jsonify({"error": "Body must be valid JSON"}), 400
    features = body.get("features")
    if not isinstance(features, list) or len(features) != 4:
        return jsonify({
            "error": "Expected 'features' to be a list of 4 numbers",
            "example": {"features": [5.1, 3.5, 1.4, 0.2]},
        }), 400
    try:
        values = [float(v) for v in features]
    except (TypeError, ValueError):
        return jsonify({"error": "All features must be numbers"}), 400
    prediction = int(model.predict([values])[0])
    return jsonify({
        "prediction": prediction,
        "class_name": CLASS_NAMES[prediction],
        "features": values,
    }), 200
if __name__ == "__main__":
    # Local testing only. Gunicorn runs the app on the EC2 instance.
    app.run(host="0.0.0.0", port=5000, debug=True)