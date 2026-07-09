import sys
import os
import json
from time import sleep

# Add edel to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.cloud import storage
from edel.io.llm import get_llm_client

def main():
    config = {
        "provider": "gemini",
        "model": "gemini-3-flash-preview",
        "location": "global"
    }
    
    print(f"Initializing GeminiClient with model: {config['model']}")
    client = get_llm_client(config)
    
    # 1. Create JSONL
    requests = [
        {"contents": [{"role": "user", "parts": [{"text": "Hello, I am request 1."}]}]},
        {"contents": [{"role": "user", "parts": [{"text": "Hello, I am request 2."}]}]}
    ]
    jsonl_content = "\n".join([json.dumps(req) for req in requests])
    
    bucket_name = "edel-batch-staging-windy-skyline-494616-m1"
    file_name = "test_input.jsonl"
    
    # 2. Upload to GCS
    storage_client = storage.Client(project="windy-skyline-494616-m1")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"inputs/{file_name}")
    blob.upload_from_string(jsonl_content)
    gcs_uri = f"gs://{bucket_name}/inputs/{file_name}"
    
    print(f"Uploaded input to {gcs_uri}")
    
    # 3. Create Batch
    try:
        from google.genai.types import CreateBatchJobConfig
        job = client.client.batches.create(
            model=client.model,
            src=gcs_uri,
            config=CreateBatchJobConfig(
                dest=f"gs://{bucket_name}/outputs/test1/"
            )
        )
        print(f"Batch created successfully! Job name: {job.name}")
        
        while True:
            status = client.client.batches.get(name=job.name)
            print(f"Job state: {status.state}")
            if status.state in ["SUCCEEDED", "FAILED", "CANCELLED"]:
                break
            sleep(10)
            
    except Exception as e:
        print(f"Failed to create batch: {e}")

if __name__ == "__main__":
    main()
