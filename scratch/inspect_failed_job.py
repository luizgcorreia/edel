import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google import genai

def main():
    # Initialize client for Vertex AI
    client = genai.Client(
        vertexai=True,
        project="windy-skyline-494616-m1",
        location="global"
    )
    
    batch_name = "projects/741561970426/locations/global/batchPredictionJobs/8387904278198484992"
    
    print(f"Fetching job: {batch_name}")
    try:
        job = client.batches.get(name=batch_name)
        print(f"Job state: {job.state}")
        
        # In the google-genai SDK, job might have an error field.
        if hasattr(job, 'error'):
            print(f"Job error: {job.error}")
        else:
            print("No top-level error field found. Let's dump the job attributes.")
            print(vars(job))
            
    except Exception as e:
        print(f"Failed to fetch job: {e}")

if __name__ == "__main__":
    main()
