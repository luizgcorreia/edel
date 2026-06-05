"""Modular LLM client interface and implementations."""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any

from dotenv import load_dotenv

load_dotenv()


class LLMClient(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response for a single prompt."""
        pass

    @abstractmethod
    def generate_embedding(self, text: str, **kwargs) -> list[float]:
        """Generate a single text embedding."""
        pass

    @abstractmethod
    def create_batch(
        self, prompts_with_ids: dict[str, str], endpoint: str = "/v1/chat/completions", **kwargs
    ) -> str:
        """Create a batch job for processing multiple requests."""
        pass

    @abstractmethod
    def poll_batch(self, batch_id: str) -> dict[str, Any]:
        """Check status of a batch job and return results if finished."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI implementation using the official SDK."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", "no-key"),
            base_url=base_url,
        )

    def generate(self, prompt: str, **kwargs) -> str:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 1),
        }
        if kwargs.get("response_format"):
            params["response_format"] = kwargs["response_format"]

        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content

    def generate_embedding(self, text: str, **kwargs) -> list[float]:
        response = self.client.embeddings.create(
            input=[text.replace("\n", " ")],
            model=self.model,
            **kwargs
        )
        return response.data[0].embedding

    def create_batch(
        self, prompts_with_ids: dict[str, str], endpoint: str = "/v1/chat/completions", **kwargs
    ) -> str:
        """Create a batch job for OpenAI."""
        import tempfile

        # Prepare JSONL
        fd, tmp_path = tempfile.mkstemp(suffix=".jsonl")
        try:
            with os.fdopen(fd, "w") as f:
                for custom_id, content in prompts_with_ids.items():
                    if endpoint == "/v1/embeddings":
                        body = {"model": self.model, "input": content.replace("\n", " ")}
                    else:
                        body = {
                            "model": self.model,
                            "messages": [{"role": "user", "content": content}],
                            "temperature": kwargs.get("temperature", 1),
                            "response_format": {"type": "json_object"},
                        }
                    
                    line = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": endpoint,
                        "body": body,
                    }
                    f.write(json.dumps(line) + "\n")

            # Upload
            with open(tmp_path, "rb") as f:
                file_batch = self.client.files.create(file=f, purpose="batch")

            # Create job
            job = self.client.batches.create(
                input_file_id=file_batch.id,
                endpoint=endpoint,
                completion_window="24h",
            )
            return job.id
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def poll_batch(self, batch_id: str) -> dict[str, Any]:
        """Poll OpenAI for batch results."""
        job = self.client.batches.retrieve(batch_id)
        
        status_info = {
            "id": job.id,
            "status": job.status,
            "request_counts": {
                "completed": job.request_counts.completed if job.request_counts else 0,
                "total": job.request_counts.total if job.request_counts else 0,
            },
            "results": None
        }

        if job.status == "completed" and job.output_file_id:
            import tempfile
            import os
            
            # Use raw response to stream bytes to a temp file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_path = tmp_file.name
                
            try:
                # Stream the download using SDK's write_to_file
                response = self.client.files.content(job.output_file_id)
                response.write_to_file(tmp_path)
                
                # Parse the file line by line
                results = {}
                with open(tmp_path, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        custom_id = data.get("custom_id")
                        if data.get("response") and data["response"].get("status_code") == 200:
                            body = data["response"]["body"]
                            if "choices" in body:
                                results[custom_id] = body["choices"][0]["message"]["content"]
                            elif "data" in body:
                                results[custom_id] = body["data"][0]["embedding"]
                            else:
                                results[custom_id] = body
                        else:
                            results[custom_id] = json.dumps({"error": "batch_failed", "details": data.get("error")})
                status_info["results"] = results
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        
        return status_info




class LMStudioClient(OpenAIClient):
    """LM Studio implementation (OpenAI-compatible)."""

    def __init__(self, model: str = "local-model", base_url: str | None = None, **kwargs):
        super().__init__(
            model=model,
            api_key="lm-studio",
            base_url=base_url or os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
            **kwargs
        )

    def create_batch(self, prompts_with_ids: dict[str, str], **kwargs) -> str:
        raise NotImplementedError("Batch API is not supported by LM Studio.")

    def poll_batch(self, batch_id: str) -> dict[str, Any]:
        raise NotImplementedError("Batch API is not supported by LM Studio.")


class GeminiClient(LLMClient):
    """Google Gemini implementation using the official google-genai SDK."""

    def __init__(self, model: str = "gemini-2.5-flash", **kwargs):
        import os
        import json
        from google import genai
        from google.genai import types
        import google.auth
        
        self.model = model
        
        # Priority: explicit config > env var > ADC project
        project_id = kwargs.get("project")
        location = kwargs.get("location", "global")
        
        if not project_id:
            try:
                _, project_id = google.auth.default()
            except Exception:
                project_id = None
        
        # Fallback to parsing ADC file for quota_project_id
        if not project_id:
            adc_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if not adc_path:
                adc_path = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
            if os.path.exists(adc_path):
                try:
                    with open(adc_path, "r") as f:
                        adc_data = json.load(f)
                        project_id = adc_data.get("quota_project_id")
                except Exception:
                    pass
        # Increase timeout for large batch uploads (google-genai expects milliseconds)
        http_options = types.HttpOptions(timeout=600.0 * 1000)
                    
        if project_id:
            # If global is requested, we use the global endpoint
            # We use v1beta1 as it is the standard for Vertex AI regional/global beta
            print(f"Initializing Gemini on Vertex AI (project: {project_id}, location: {location})")
            self.client = genai.Client(
                vertexai=True, 
                project=project_id, 
                location=location,
                http_options=http_options
            )
            self.project_id = project_id
            self.location = location
        else:
            # Fallback to default API key mode
            self.client = genai.Client(http_options=http_options)
            self.project_id = None
            self.location = None

    def generate(self, prompt: str, **kwargs) -> str:
        from google.genai import types
        
        # Enforce JSON output as requested by the original design
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=kwargs.get("temperature", 1.0),
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config
        )
        return response.text

    def generate_embedding(self, text: str, **kwargs) -> list[float]:
        # Using the standard embedding model for Gemini
        response = self.client.models.embed_content(
            model="text-embedding-004",
            contents=text,
        )
        return response.embeddings[0].values

    def _get_or_create_bucket(self):
        from google.cloud import storage
        from google.api_core.exceptions import Conflict
        
        if not self.project_id:
            raise ValueError("project_id is required for Vertex AI batch processing.")
            
        bucket_name = f"edel-batch-staging-{self.project_id}".lower().replace("_", "-")
        client = storage.Client(project=self.project_id)
        
        try:
            client.get_bucket(bucket_name)
        except Exception:
            try:
                bucket = client.bucket(bucket_name)
                # Location must be a region for Vertex AI, or 'US'/'EU'. 
                bucket.location = self.location if self.location != 'global' else 'us-central1'
                client.create_bucket(bucket)
            except Conflict:
                pass
        return bucket_name, client

    def create_batch(self, prompts_with_ids: dict[str, str], **kwargs) -> str:
        import json
        import uuid
        import hashlib
        from google.genai.types import CreateBatchJobConfig
        
        bucket_name, storage_client = self._get_or_create_bucket()
        job_uuid = str(uuid.uuid4())
        
        # 1. Create JSONL and content-hash based ID Map
        # NOTE: Vertex AI Batch API does NOT guarantee output order matches input.
        # We use a content hash of each prompt to build a reverse lookup so that
        # output lines (which include the echoed request) can be matched back to
        # their original custom_id regardless of shuffling.
        requests = []
        id_map = {}
        is_embedding = "embedding" in self.model or kwargs.get("endpoint") == "/v1/embeddings"
        
        for idx, (custom_id, prompt) in enumerate(prompts_with_ids.items()):
            if is_embedding:
                requests.append({
                    "request": {
                        "content": {"parts": [{"text": prompt}]}
                    }
                })
            else:
                requests.append({
                    "request": {
                        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
                    }
                })
            # Key by content hash instead of positional index
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            id_map[prompt_hash] = custom_id
            
        jsonl_content = "\n".join([json.dumps(req) for req in requests])
        
        # 2. Upload to GCS
        bucket = storage_client.bucket(bucket_name)
        input_blob = bucket.blob(f"batch_jobs/{job_uuid}/input.jsonl")
        input_blob.upload_from_string(jsonl_content, timeout=600.0)
        
        map_blob = bucket.blob(f"batch_jobs/{job_uuid}/id_map.json")
        map_blob.upload_from_string(json.dumps(id_map))
        
        gcs_uri = f"gs://{bucket_name}/batch_jobs/{job_uuid}/input.jsonl"
        dest_prefix = f"gs://{bucket_name}/batch_jobs/{job_uuid}/output/"
        
        # 3. Create Job
        job = self.client.batches.create(
            model=self.model,
            src=gcs_uri,
            config=CreateBatchJobConfig(dest=dest_prefix)
        )
        return f"{job.name}::{job_uuid}::{len(prompts_with_ids)}"

    def poll_batch(self, batch_id_composite: str) -> dict[str, Any]:
        import json
        import hashlib
        parts = batch_id_composite.split("::")
        batch_id = parts[0]
        job_uuid = parts[1] if len(parts) > 1 else ""
        total_requests = int(parts[2]) if len(parts) > 2 else 0
        
        job = self.client.batches.get(name=batch_id)
        state_str = str(job.state).upper()
        
        status_info = {
            "id": batch_id_composite,
            "status": "in_progress",
            "request_counts": {
                "completed": 0, # Vertex API doesn't expose partial progress easily
                "total": total_requests,
            },
            "results": None
        }
        
        if "SUCCEEDED" in state_str:
            status_info["status"] = "completed"
            status_info["request_counts"]["completed"] = total_requests
        elif "FAILED" in state_str or "CANCELLED" in state_str:
            status_info["status"] = "failed"
            return status_info
        else:
            return status_info
            
        if status_info["status"] == "completed":
            bucket_name, storage_client = self._get_or_create_bucket()
            bucket = storage_client.bucket(bucket_name)
            
            # Download id_map (keyed by content hash)
            map_blob = bucket.blob(f"batch_jobs/{job_uuid}/id_map.json")
            id_map = json.loads(map_blob.download_as_string())
            
            # Detect id_map format: legacy (positional) vs new (content-hash)
            # Legacy maps have small integer string keys like "0", "1", "2"
            sample_keys = list(id_map.keys())[:3]
            is_legacy_map = all(k.isdigit() and len(k) < 10 for k in sample_keys)
            
            if is_legacy_map:
                # Legacy format: need to download input.jsonl to rebuild
                # content-hash lookup from the original prompts
                input_blob = bucket.blob(f"batch_jobs/{job_uuid}/input.jsonl")
                input_content = input_blob.download_as_string().decode("utf-8")
                input_lines = [l for l in input_content.split("\n") if l.strip()]
                
                hash_to_custom_id = {}
                for i, input_line in enumerate(input_lines):
                    inp_data = json.loads(input_line)
                    req = inp_data.get("request", {})
                    # Extract prompt text from either generation or embedding format
                    if "contents" in req:
                        prompt_text = req["contents"][0]["parts"][0]["text"]
                    elif "content" in req:
                        prompt_text = req["content"]["parts"][0]["text"]
                    else:
                        continue
                    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
                    custom_id = id_map.get(str(i))
                    if custom_id:
                        hash_to_custom_id[prompt_hash] = custom_id
            else:
                # New format: id_map is already keyed by content hash
                hash_to_custom_id = id_map
            
            # Download output files and match by content hash
            results = {}
            blobs = bucket.list_blobs(prefix=f"batch_jobs/{job_uuid}/output/")
            for blob in blobs:
                if blob.name.endswith(".jsonl"):
                    content = blob.download_as_string().decode('utf-8')
                    lines = [line for line in content.split('\n') if line.strip()]
                    for line in lines:
                        try:
                            data = json.loads(line)
                            res = data.get("response", {})
                            
                            # Extract prompt from the echoed request to compute hash
                            req = data.get("request", {})
                            if "contents" in req:
                                prompt_text = req["contents"][0]["parts"][0]["text"]
                            elif "content" in req:
                                prompt_text = req["content"]["parts"][0]["text"]
                            else:
                                continue
                            prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
                            custom_id = hash_to_custom_id.get(prompt_hash)
                            if not custom_id:
                                continue
                            
                            # Check if it's an embedding response
                            if "embeddings" in res and res["embeddings"]:
                                parsed_val = res["embeddings"][0].get("values", [])
                            else:
                                # Standard generation response
                                candidates = res.get("candidates", [])
                                if candidates and candidates[0].get("content", {}).get("parts"):
                                    parsed_val = candidates[0]["content"]["parts"][0].get("text", "{}")
                                else:
                                    parsed_val = "{}"
                                
                            results[custom_id] = parsed_val
                        except Exception as e:
                            print(f"Error parsing batch output line: {e}")
                            
            status_info["results"] = results
            
        return status_info


class MockClient(LLMClient):
    """Mock client for testing."""

    def __init__(self, response: str | None = None, **kwargs):
        self.batches = {}
        self.response = response

    def generate(self, prompt: str, **kwargs) -> str:
        if self.response:
            return self.response

        return json.dumps(
            {
                "problem": "Mock problem",
                "method": "Mock method",
                "finding": "Mock finding",
                "interpretation": "Mock interpretation",
                "cluster_topics": "Mock topics",
                "proposed_label": "Mock label",
                "axis_label": "Mock axis",
                "negative_pole": "Mock neg",
                "positive_pole": "Mock pos",
            }
        )

    def generate_embedding(self, text: str, **kwargs) -> list[float]:
        import numpy as np

        # Default to 1536 (ada-002) if not specified
        dim = kwargs.get("dimensions", 1536)
        return np.random.rand(dim).tolist()

    def create_batch(
        self, prompts_with_ids: dict[str, str], endpoint: str = "/v1/chat/completions", **kwargs
    ) -> str:
        batch_id = f"mock-batch-{time.time()}"
        self.batches[batch_id] = (prompts_with_ids, endpoint)
        return batch_id

    def poll_batch(self, batch_id: str) -> dict[str, Any]:
        data = self.batches.get(batch_id)
        if not data:
            return {"id": batch_id, "status": "failed", "results": None}
        
        prompts, endpoint = data
        
        if endpoint == "/v1/embeddings":
            results = {cid: self.generate_embedding("") for cid in prompts.keys()}
        else:
            results = {cid: self.generate("") for cid in prompts.keys()}

        return {
            "id": batch_id,
            "status": "completed",
            "request_counts": {"completed": len(prompts), "total": len(prompts)},
            "results": results,
        }


class NullClient(LLMClient):
    """Null provider that slices abstracts into four equal segments."""

    def __init__(self, **kwargs):
        self.batches = {}

    def generate(self, prompt: str, **kwargs) -> str:
        import re
        abstract_text = ""
        # Search for "Abstract:" case-insensitively followed by optional spaces and newline
        match_start = re.search(r'(?i)\babstract:\s*\n', prompt)
        if match_start:
            content_from_abstract = prompt[match_start.end():]
            # Find where the abstract ends - usually at "Keywords:" or "JSON Answer:" or "Rules:"
            match_end = re.search(r'(?i)\n(?:keywords|json answer|rules):\s*\n', content_from_abstract)
            if match_end:
                abstract_text = content_from_abstract[:match_end.start()].strip()
            else:
                # If no ending marker found, take up to "JSON Answer:" or "Keywords:" without requiring the exact format
                match_end_fallback = re.search(r'(?i)\n(?:keywords|json answer)\b', content_from_abstract)
                if match_end_fallback:
                    abstract_text = content_from_abstract[:match_end_fallback.start()].strip()
                else:
                    abstract_text = content_from_abstract.strip()
        else:
            # Fallback: if we can't find the section, use the prompt itself
            abstract_text = prompt.strip()

        n = len(abstract_text)
        w1 = n // 4
        w2 = (2 * n) // 4
        w3 = (3 * n) // 4

        data = {
            "problem": abstract_text[:w1].strip(),
            "method": abstract_text[w1:w2].strip(),
            "finding": abstract_text[w2:w3].strip(),
            "interpretation": abstract_text[w3:].strip()
        }
        return json.dumps(data)

    def generate_embedding(self, text: str, **kwargs) -> list[float]:
        import numpy as np
        dim = kwargs.get("dimensions", 1536)
        return np.random.rand(dim).tolist()

    def create_batch(
        self, prompts_with_ids: dict[str, str], endpoint: str = "/v1/chat/completions", **kwargs
    ) -> str:
        batch_id = f"null-batch-{time.time()}"
        self.batches[batch_id] = (prompts_with_ids, endpoint)
        return batch_id

    def poll_batch(self, batch_id: str) -> dict[str, Any]:
        data = self.batches.get(batch_id)
        if not data:
            return {"id": batch_id, "status": "failed", "results": None}
        
        prompts, endpoint = data
        results = {cid: self.generate(prompt) for cid, prompt in prompts.items()}
        return {
            "id": batch_id,
            "status": "completed",
            "request_counts": {"completed": len(prompts), "total": len(prompts)},
            "results": results,
        }


def get_llm_client(config: dict) -> LLMClient:
    """Factory to create the appropriate LLM client."""
    provider = config.get("provider", "openai")
    model = config.get("model", "gpt-4o-mini")

    if provider == "openai":
        return OpenAIClient(**config)
    elif provider == "lmstudio":
        return LMStudioClient(**config)
    elif provider == "gemini":
        return GeminiClient(**config)
    elif provider == "mock":
        return MockClient()
    elif provider == "null":
        return NullClient(**config)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
