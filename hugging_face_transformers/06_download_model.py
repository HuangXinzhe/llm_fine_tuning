from huggingface_hub import snapshot_download

snapshot_download(repo_id="bert-base-cased",
                  local_dir="./model/bert-base-cased")
