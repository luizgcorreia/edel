import asyncio
from google import genai
from google.genai import types

def test_model(model_name, location):
    print(f"Testing model {model_name} in location {location}...")
    try:
        client = genai.Client(vertexai=True, location=location, project="windy-skyline-494616-m1")
        job = client.batches.create(
            model=model_name,
            src="gs://edel-embeddings-testing/dummy.jsonl",
            config=types.CreateBatchJobConfig(dest="gs://edel-embeddings-testing/")
        )
        print(f"SUCCESS: Batch job created for {model_name} in {location}")
        job.delete()
    except Exception as e:
        print(f"ERROR for {model_name} in {location}: {e}")

test_model("gemini-embedding-001", "us-central1")
test_model("gemini-embedding-001", "global")
test_model("text-embedding-004", "us-central1")
test_model("text-embedding-005", "us-central1")
test_model("text-embedding-005", "global")
