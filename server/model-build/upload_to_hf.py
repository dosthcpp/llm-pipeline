#!/usr/bin/env python3
"""Defensive, resumable uploader for custom models -> Hugging Face Hub.

TON-2391 (model build research). Handles the lab's GGUF/artifact case and the
full-folder (safetensors) case. Auth via HF_TOKEN env (never hard-coded).

Examples:
    # Preview what would happen, no network writes:
    HF_TOKEN=hf_xxx python upload_to_hf.py \
        --repo tony-ai-lab/qwopus-mtp-q4km \
        --path ./Qwopus-MTP-Q4_K_M.gguf --private --dry-run

    # Real upload of a single GGUF file:
    HF_TOKEN=hf_xxx python upload_to_hf.py \
        --repo tony-ai-lab/qwopus-mtp-q4km \
        --path ./Qwopus-MTP-Q4_K_M.gguf --private

    # Resumable upload of a whole folder of weights:
    HF_TOKEN=hf_xxx python upload_to_hf.py \
        --repo tony-ai-lab/my-finetune \
        --path ./out --large --private
"""
import argparse
import os
import sys
from pathlib import Path


def die(msg: str, code: int = 1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def main():
    ap = argparse.ArgumentParser(description="Upload a custom model to the HF Hub.")
    ap.add_argument("--repo", required=True, help="repo_id, e.g. namespace/name")
    ap.add_argument("--path", required=True, help="local file or folder to upload")
    ap.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    ap.add_argument("--path-in-repo", default=None,
                    help="destination path inside the repo (single-file/folder only)")
    ap.add_argument("--private", action="store_true", help="create the repo private")
    ap.add_argument("--large", action="store_true",
                    help="use resumable upload_large_folder (folders, many/large files)")
    ap.add_argument("--commit-message", default="Upload custom model (TON-2391)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate + print plan, do not create repo or upload")
    args = ap.parse_args()

    src = Path(args.path).expanduser()
    if not src.exists():
        die(f"local path does not exist: {src}")

    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        die("HF_TOKEN env var not set. Provision a WRITE token from "
            "hf.co/settings/tokens and export HF_TOKEN before a real run.")

    is_dir = src.is_dir()
    size = (sum(f.stat().st_size for f in src.rglob('*') if f.is_file())
            if is_dir else src.stat().st_size)
    print("== upload plan ==")
    print(f"  repo_id     : {args.repo} ({args.repo_type}, "
          f"{'private' if args.private else 'public'})")
    print(f"  local source: {src}  [{'folder' if is_dir else 'file'}, "
          f"{size/1e9:.2f} GB]")
    print(f"  mode        : {'upload_large_folder' if (is_dir and args.large) else ('upload_folder' if is_dir else 'upload_file')}")
    if size > 5e9 and is_dir and not args.large:
        print("  HINT        : >5GB folder — consider --large (resumable).")

    if args.dry_run:
        print("== dry-run: no changes made ==")
        return

    # Speed knob for big weights; safe to always set.
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        die('huggingface_hub not installed. Run: pip install -U "huggingface_hub"')

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo, repo_type=args.repo_type,
                    private=args.private, exist_ok=True)
    print(f"  repo ready  : https://huggingface.co/{args.repo}")

    if is_dir and args.large:
        api.upload_large_folder(repo_id=args.repo, repo_type=args.repo_type,
                                folder_path=str(src))
    elif is_dir:
        api.upload_folder(repo_id=args.repo, repo_type=args.repo_type,
                          folder_path=str(src), path_in_repo=args.path_in_repo,
                          commit_message=args.commit_message)
    else:
        api.upload_file(repo_id=args.repo, repo_type=args.repo_type,
                        path_or_fileobj=str(src),
                        path_in_repo=args.path_in_repo or src.name,
                        commit_message=args.commit_message)
    print(f"== done: https://huggingface.co/{args.repo} ==")


if __name__ == "__main__":
    main()
