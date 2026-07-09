import sys
import os
from time import sleep

# Add edel to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edel.io.llm import get_llm_client
from google.genai.types import CreateBatchJobConfig

def main():
    config = {
        "provider": "gemini",
        "model": "gemini-3.1-pro",
        "location": "global"
    }
    
    print(f"Initializing GeminiClient with model: {config['model']}")
    client = get_llm_client(config)
    
    # Try inline requests
    inline_requests = [
        {"contents": [{"parts": [{"text": "Hello 1"}]}]},
        {"contents": [{"parts": [{"text": "Hello 2"}]}]}
    ]
    
    try:
        print("Creating batch job...")
        job = client.client.batches.create(
            model=client.model,
            src=inline_requests
        )
        print(f"Batch created successfully! Job name: {job.name}")
        
        # We'll immediately delete it or cancel it if we can
    except Exception as e:
        print(f"Failed to create inline batch: {e}")

if __name__ == "__main__":
    main()
