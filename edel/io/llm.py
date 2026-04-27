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
            content = self.client.files.content(job.output_file_id).text
            results = {}
            for line in content.strip().split("\n"):
                data = json.loads(line)
                custom_id = data.get("custom_id")
                if data.get("response") and data["response"].get("status_code") == 200:
                    body = data["response"]["body"]
                    if "choices" in body:
                        results[custom_id] = body["choices"][0]["message"]["content"]
                    elif "data" in body:
                        # Return the embedding list directly
                        results[custom_id] = body["data"][0]["embedding"]
                    else:
                        results[custom_id] = body
                else:
                    results[custom_id] = json.dumps({"error": "batch_failed", "details": data.get("error")})
            status_info["results"] = results
        
        return status_info


class LMStudioClient(OpenAIClient):
    """LM Studio implementation (OpenAI-compatible)."""

    def __init__(self, model: str = "local-model", base_url: str | None = None):
        super().__init__(
            model=model,
            api_key="lm-studio",
            base_url=base_url or os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
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
                    
        if project_id:
            # If global is requested, we use the global endpoint
            # We use v1beta1 as it is the standard for Vertex AI regional/global beta
            print(f"Initializing Gemini on Vertex AI (project: {project_id}, location: {location})")
            self.client = genai.Client(
                vertexai=True, 
                project=project_id, 
                location=location
            )
        else:
            # Fallback to default API key mode
            self.client = genai.Client()

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

    def create_batch(self, prompts_with_ids: dict[str, str], **kwargs) -> str:
        raise NotImplementedError("Batch API is not natively supported by this GeminiClient implementation.")

    def poll_batch(self, batch_id: str) -> dict[str, Any]:
        raise NotImplementedError("Batch API is not natively supported by this GeminiClient implementation.")


class MockClient(LLMClient):
    """Mock client for testing."""

    def __init__(self, response: str | None = None):
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
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
