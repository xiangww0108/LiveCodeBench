"""
Upload fine-tuned planner model to HuggingFace Hub
"""

from huggingface_hub import HfApi, create_repo
import os

def upload_model_to_hf(local_dir, repo_id, token):
    """
    Upload the fine-tuned model to HuggingFace Hub
    
    Args:
        local_dir: Local directory containing the model
        repo_id: HuggingFace repo ID (e.g., "username/repo-name")
        token: HuggingFace API token
    """
    print(f"Uploading model from {local_dir} to {repo_id}...")
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, token=token, repo_type="model", exist_ok=True)
        print(f"Repository {repo_id} ready")
    except Exception as e:
        print(f"Note: {e}")
    
    # Initialize API
    api = HfApi()
    
    # Upload all files in the directory
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
        token=token
    )
    
    print(f"✓ Model successfully uploaded to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace")
    parser.add_argument("--model_dir", type=str, default="./planner-finetuned",
                        help="Local directory containing the model")
    parser.add_argument("--repo_id", type=str, default="Intellegen4/planner-qwen2.5-4b",
                        help="HuggingFace repository ID")
    parser.add_argument("--token", type=str, required=True,
                        help="HuggingFace API token")
    
    args = parser.parse_args()
    
    upload_model_to_hf(args.model_dir, args.repo_id, args.token)