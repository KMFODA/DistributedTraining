# upload_worker.py
import sys
from huggingface_hub import upload_folder

if __name__ == "__main__":
    repo_id = sys.argv[1]
    local_dir = sys.argv[2]
    commit_message = sys.argv[3]

    upload_folder(
        repo_id=repo_id,
        folder_path=local_dir,
        commit_message=commit_message,
    )
