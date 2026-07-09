import sys
import os
from time import sleep

# Add edel to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edel.io.llm import get_llm_client

def main():
    config = {
        "provider": "gemini",
        "model": "gemini-3-flash-preview",
        "location": "global"
    }
    
    print(f"Initializing GeminiClient with model: {config['model']}")
    client = get_llm_client(config)
    
    prompts = {
        "req-001": "Write a short poem about data processing.",
        "req-002": "What is 2+2?"
    }
    
    try:
        print("Creating batch job via GeminiClient...")
        batch_id = client.create_batch(prompts)
        print(f"Batch created successfully! Composite ID: {batch_id}")
        
        print("Polling batch status...")
        while True:
            status_info = client.poll_batch(batch_id)
            print(f"Status: {status_info['status']}")
            
            if status_info["status"] in ["completed", "failed"]:
                if status_info["status"] == "completed":
                    print("\nResults:")
                    for cid, text in status_info["results"].items():
                        print(f"[{cid}]: {text}")
                break
            sleep(10)
            
    except Exception as e:
        print(f"Failed to create/poll batch: {e}")

if __name__ == "__main__":
    main()
