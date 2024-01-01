from huggingface_hub import snapshot_download

snapshot_download(repo_id="bert-base-chinese", cache_dir="./cache_dir")
