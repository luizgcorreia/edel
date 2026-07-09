from edel.io.llm import OpenAIClient

def main():
    client = OpenAIClient()
    file_id = "file-EzaL3UvWHNA7tmYPVLJ25L"
    dest_path = "artifacts/embeddings/openalex_T10102_global/batch_output.jsonl"
    
    print(f"Downloading {file_id} to {dest_path} using write_to_file...")
    response = client.client.files.content(file_id)
    response.write_to_file(dest_path)
    print("Download completed successfully!")

if __name__ == "__main__":
    main()
