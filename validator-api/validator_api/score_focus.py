import os, json
import asyncio
from datetime import datetime
from typing import Tuple
import boto3
from botocore.exceptions import ClientError
import google.generativeai as genai

from validator_api import config
from omega.imagebind_wrapper import run_async

from validator_api.database.models.focus_video_record import (
    FocusVideoRecord, FocusVideoInternal, FocusVideoStateInternal,
    FocusVideoExternal, FocusVideoExternalWithComputed
)
from validator_api.database import get_db_context

GOOGLE_AI_API_KEY = config.GOOGLE_AI_API_KEY
genai.configure(api_key=GOOGLE_AI_API_KEY)

genai_model = genai.GenerativeModel('gemini-1.5-pro')

s3_client = boto3.client(
    's3',
    aws_access_key_id = config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key = config.AWS_SECRET_ACCESS_KEY,
    region_name=config.AWS_S3_REGION
)

def get_s3_path(video_id: str) -> str:
    return f"clips/{video_id}.webm"

# from services.gemini_service import calculate_gemini_score
# from services.video_embedding_service import calculate_video_embedding_similarity
# from services.text_embedding_service import calculate_text_embedding
# from services.tao_service import calculate_expected_reward_tao, get_tao_usd_rate

async def calculate_focus_score(video_id: str, focusing_task: str) -> Tuple[float, dict]:
    # FV TODO: If the max_focus_tao has been hit already, mark the video as REJECTED, or maybe
    # we don't reject the video and let it go to the marketplace but miners just can't purchase it

    #gemini_score = await calculate_gemini_score(video_url, focusing_task)
    #video_embedding_similarity = await calculate_video_embedding_similarity(video_url)
    #text_embedding = await calculate_text_embedding(focusing_task)
    
    #video_score = (gemini_score + video_embedding_similarity + text_embedding) / 3
    #video_details = f"Gemini Score: {gemini_score}, Video Embedding Similarity: {video_embedding_similarity}, Text Embedding: {text_embedding}"

    try:
        object_name = get_s3_path(video_id)
        print(f"Attempting to download object: {object_name}")
        print(f"From bucket: {config.AWS_S3_BUCKET_NAME}")
        
        # Create the .focus_videos directory if it doesn't exist
        os.makedirs('.focus_videos', exist_ok=True)
        file_name = os.path.join('.focus_videos', os.path.basename(object_name))
        
        # Add error handling to the download_file call
        try:
            result = await run_async(s3_client.download_file, config.AWS_S3_BUCKET_NAME, object_name, file_name)
            if os.path.exists(file_name): 
                print(f"Download succeeded: {file_name}")
            else: 
                print(f"Download failed: File not found locally after download attempt")
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                print(f"Error: The object {object_name} was not found in the bucket {config.AWS_S3_BUCKET_NAME}")
            else:
                print(f"Unexpected error: {e}")
        
        video_file = await run_async(genai.upload_file, file_name)
        jsonSchema = {
            'title': "score video productivity",
            'type': 'object',
            'properties': {
                'score': {
                    'type': 'float',
                    'description': "score value of video productivity from 0 to 1"
                },
                'analysis': {
                   'type': 'string',
                   'description': 'analysis of video productivity'
                }
            }
        }
        while video_file.state.name == "PROCESSING":
            print('Uploading video processing..')
            await asyncio.sleep(1)
            video_file = await run_async(genai.get_file, video_file.name)
            
        prompt = f"""
            Score the productivity of the attached video based on whether it shows genuine work on the focusing_task.
            - Score 0.00 if the video content does not align with or is unrelated to the focusing_task.
            - Score 1.00 if the video clearly shows focused work on the focusing_task, including relevant actions and context.
            - Provide a score between 0.00 and 1.00, formatted to two decimal places (e.g., 0.73), using intervals of 0.01.
            focusing_task is {focusing_task}.
            Ensure the video is analyzed for contextual relevance, activity, and alignment with the task.
            The response has to be only number as float type(score).
        """

        contents = [
            video_file,
            prompt
        ]

        response = await run_async(genai_model.generate_content, contents)
        #print(f'score: {response.text} {video_file.state.name}')
        video_score = float(response.text)
        video_details = {
            "description": "This is a random score, testing purposes only",
            "focusing_task": focusing_task,
        }

        with get_db_context() as db:
            record = db.query(FocusVideoRecord).filter(FocusVideoRecord.video_id == video_id).first()

            assert record is not None, "Focus video record not found"
            if record.processing_state != FocusVideoStateInternal.PROCESSING:
                print(f"ERROR!!!!! NOTIFY SALMAN! Focus video record {video_id} is not in PROCESSING state")
            assert record.deleted_at is None, "Focus video record is deleted"

            if record:
                record.processing_state = FocusVideoStateInternal.READY
                record.updated_at = datetime.utcnow()
                record.video_score = video_score
                record.video_details = {**(record.video_details or {}), **video_details}
                db.commit()
                print("Updated focus video record in the database")

        return video_score, video_details

    except Exception as e:
       print(e)
       with get_db_context() as db:
            record = db.query(FocusVideoRecord).filter(FocusVideoRecord.video_id == video_id).first()
            if record:
                record.processing_state = FocusVideoStateInternal.REJECTED
                record.updated_at = datetime.utcnow()
                record.rejection_reason = str(e)
                db.commit()
    
    finally:
        if os.path.exists(file_name):
            try:
                os.remove(file_name)
            except Exception as e:
                print(f"Error while trying to remove file {file_name}: {e}")

    
if __name__ == "__main__":
    video_id = "1e268a62-d4fc-4157-afd3-274a423a0c5c"
    focusing_task = "Talking to the camera about how technology is an unstoppable force."
    score = asyncio.run(calculate_focus_score(video_id, focusing_task))
    print(score)
