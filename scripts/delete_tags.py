from huggingface_hub import list_repo_refs, delete_tag

repo_id = "kmfoda/gpt2-1b"

refs = list_repo_refs(repo_id, repo_type="model")
tags = refs.tags if refs.tags else None

if tags != None:
    for tag in tags:
        if tag.name == "0":
            continue
        else:
            delete_tag(repo_id, tag=tag.name)
