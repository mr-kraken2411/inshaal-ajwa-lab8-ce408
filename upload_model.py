"""Upload the trained model to S3."""
import sys
import boto3
from botocore.exceptions import ClientError
BUCKET_NAME = "ml-iris-model-inshaal-ajwa"   # <-- your bucket name
MODEL_FILE = "iris_model.pkl"
s3 = boto3.client("s3")
try:
    s3.upload_file(MODEL_FILE, BUCKET_NAME, MODEL_FILE)
    print("Uploaded {} to s3://{}/{}".format(MODEL_FILE, BUCKET_NAME, MODEL_FILE))
except FileNotFoundError:
    print("{} not found. Run train_model.py first.".format(MODEL_FILE))
    sys.exit(1)
except ClientError as e:
    print("Upload failed: {}".format(e))
    sys.exit(1)