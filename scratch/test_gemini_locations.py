import os
import time
import json
from google import genai
import google.auth
from google.genai import types

def test_config(vertexai, location):
    print(f"\n--- Testing VertexAI={vertexai}, Location={location} ---")
    try:
        http_options = types.HttpOptions(timeout=10000.0)
        
        project_id = None
        try:
            _, project_id = google.auth.default()
        except Exception:
            pass
            
        if not project_id:
            adc_path = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
            if os.path.exists(adc_path):
                with open(adc_path, "r") as f:
                    adc_data = json.load(f)
                    project_id = adc_data.get("quota_project_id")

        if vertexai:
            client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
                http_options=http_options
            )
        else:
            client = genai.Client(http_options=http_options)

        start = time.time()
        # Test simple generate content call
        # Try different model names depending on vertex vs developer
        model_name = "gemini-2.5-flash"
        response = client.models.generate_content(
            model=model_name,
            contents="Hello!",
        )
        print(f"Success in {time.time() - start:.2f}s!")
        print("Response:", response.text)
        return True
    except Exception as e:
        print(f"Failed: {type(e).__name__}: {e}")
        return False

def main():
    print("Testing 'global' with Vertex AI...")
    test_config(vertexai=True, location="global")
    
    print("\nTesting 'us-central1' with Vertex AI...")
    test_config(vertexai=True, location="us-central1")

    print("\nTesting 'europe-west1' with Vertex AI...")
    test_config(vertexai=True, location="europe-west1")

if __name__ == "__main__":
    main()
