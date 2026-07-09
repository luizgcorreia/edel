from edel.io.llm import OpenAIClient

def main():
    client = OpenAIClient()
    file_id = "file-86PQDkFk62hWbZqLfdMiqF"
    dest_path = "artifacts/embeddings/openalex_T10102_global/batch_output_2.jsonl"
    
    print(f"Downloading batch 2 output {file_id} to {dest_path}...")
    response = client.client.files.content(file_id)
    response.write_to_file(dest_path)
    print("Download completed successfully!")

if __name__ == "__main__":
    main()
