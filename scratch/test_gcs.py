from google.cloud import storage
from google.api_core.exceptions import Conflict

def get_or_create_bucket(project_id: str, location: str) -> str:
    # Ensure bucket name is valid (lowercase, no underscores)
    bucket_name = f"edel-batch-staging-{project_id}".lower().replace("_", "-")
    client = storage.Client(project=project_id)
    
    try:
        bucket = client.get_bucket(bucket_name)
        print(f"Bucket {bucket_name} already exists.")
    except Exception:
        print(f"Bucket {bucket_name} not found. Creating...")
        try:
            bucket = client.bucket(bucket_name)
            # Location must be a region for Vertex AI, or 'US'/'EU'. 
            # We'll use the location provided (e.g. 'us-central1')
            bucket.location = location if location != 'global' else 'us-central1'
            bucket = client.create_bucket(bucket)
            print(f"Created bucket {bucket_name}")
        except Conflict:
            print(f"Bucket {bucket_name} already exists (Conflict).")
        except Exception as e:
            print(f"Error creating bucket: {e}")
            raise
    
    return bucket_name

if __name__ == "__main__":
    get_or_create_bucket("windy-skyline-494616-m1", "global")
