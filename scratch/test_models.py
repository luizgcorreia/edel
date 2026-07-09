import sys
from edel.io.llm import GeminiClient
from google.genai.types import CreateBatchJobConfig

def test_model(model_name, location):
    print(f"\n--- Testing Model: {model_name} | Location: {location} ---")
    try:
        client = GeminiClient(model=model_name, location=location)
        # Attempt a synchronous embedding call
        response = client.client.models.embed_content(
            model=model_name,
            contents="hello world"
        )
        print(f"[Sync] Success! Vector size: {len(response.embeddings[0].values)}")
    except Exception as e:
        print(f"[Sync] Error: {type(e).__name__}: {e}")
        
    try:
        # Attempt a dummy batch job creation to see if the PublisherModel is rejected
        client.client.batches.create(
            model=model_name,
            src="gs://dummy-bucket/dummy-input.jsonl",
            config=CreateBatchJobConfig(dest="gs://dummy-bucket/dummy-output/")
        )
        print("[Batch] Success (or failed later)! PublisherModel accepted.")
    except Exception as e:
        print(f"[Batch] Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    models = ["gemini-embedding-001", "text-embedding-005", "textembedding-gecko@001", "text-embedding-004"]
    locations = ["global", "us-central1"]
    
    for loc in locations:
        for mod in models:
            test_model(mod, loc)
