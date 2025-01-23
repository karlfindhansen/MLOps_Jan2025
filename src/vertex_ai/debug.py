from google.cloud import storage

client = storage.Client()
bucket_name = "bird-classification-data"  # Replace with your GCS bucket name
dataset_prefix = "files/md5"

bucket = client.bucket(bucket_name)
blobs = bucket.list_blobs(prefix=dataset_prefix)

from PIL import Image

with Image.open("data/00a0a5ea4ca864f0196034f3a8cce4") as img:
    img.show()
