import os
import sys
import json
from google import genai
import google.auth
from google.genai import types

def main():
    print("Python version:", sys.version)
    print("Checking google.auth...")
    try:
        creds, project = google.auth.default()
        print("Default credentials project:", project)
        print("Credentials type:", type(creds))
    except Exception as e:
        print("Error getting default credentials:", e)

    print("\nAttempting to initialize genai.Client...")
    try:
        # Increase timeout for handshake/requests (google-genai expects milliseconds)
        http_options = types.HttpOptions(timeout=60000.0)
        
        # Priority: explicit config > env var > ADC project
        project_id = None
        try:
            _, project_id = google.auth.default()
        except Exception:
            pass
            
        if not project_id:
            adc_path = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
            if os.path.exists(adc_path):
                try:
                    with open(adc_path, "r") as f:
                        adc_data = json.load(f)
                        project_id = adc_data.get("quota_project_id")
                except Exception:
                    pass

        if project_id:
            print(f"Using Vertex AI with project: {project_id}")
            client = genai.Client(
                vertexai=True,
                project=project_id,
                location="global",
                http_options=http_options
            )
        else:
            print("Using default API Key mode")
            client = genai.Client(http_options=http_options)

        print("Client initialized successfully.")
        
        # Test generation
        print("\nTesting generate_content...")
        model_name = "gemini-2.5-flash"
        response = client.models.generate_content(
            model=model_name,
            contents="Say hello!",
        )
        print("Response text:", response.text)

    except Exception as e:
        print("An error occurred during execution:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
