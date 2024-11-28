import os
import json
from sqlalchemy import text
from validator_api.services.scoring_service import get_gcs_uri, get_s3_path
from validator_api.database import get_db_context
from google.cloud import storage
from datetime import timedelta
from validator_api import config

videos_query = text("""
SELECT video_id, created_at, video_score, processing_state, video_details->>'boosted_multiplier', video_details->>'duration', rejection_reason
FROM focus_videos
WHERE created_at > CURRENT_DATE - INTERVAL '24 hours' AND (video_details->>'boosted_multiplier')::NUMERIC > 1.0;
""")

videos_query = text("""
SELECT video_id, created_at, video_score, processing_state, video_details->>'boosted_multiplier', video_details->>'duration', rejection_reason
FROM focus_videos
WHERE video_id = 'cHDO7MVYy';
""")

GCP_STORAGE_CLIENT = storage.Client()

def get_gcs_presigned_url(video_id: str, expiration_time_seconds: int = 3600) -> str:
    bucket = GCP_STORAGE_CLIENT.bucket(config.GOOGLE_CLOUD_BUCKET_NAME)
    blob = bucket.blob(get_s3_path(video_id))
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(seconds=expiration_time_seconds),
        method="GET",
    )
    return url

def download_file_from_gcs(key: str, file_path: str):
    """Downloads a file from the GCS bucket."""
    bucket = GCP_STORAGE_CLIENT.bucket(config.GOOGLE_CLOUD_BUCKET_NAME)
    blob = bucket.blob(key)
    blob.download_to_filename(file_path)

os.makedirs("gcs_downloads", exist_ok=True)
datapoints = []
with get_db_context() as db:
    result = db.execute(videos_query)
    results = result.fetchall()
    print(f"Found {len(results)} videos")
    for video in results:
        video_id = video[0]
        datapoints.append({
            "video_id": video_id,
            "created_at": video[1].strftime("%Y-%m-%d %H:%M:%S"),
            "video_score": video[2],
            "processing_state": video[3],
            "boosted_multiplier": video[4],
            "duration": video[5],
            "rejection_reason": video[6],
        })
        download_file_from_gcs(get_s3_path(video_id), f"gcs_downloads/{video_id}.webm")

with open("datapoints.json", "w") as f:
    json.dump(datapoints, f, indent=4)
