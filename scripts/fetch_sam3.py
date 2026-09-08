"""
Preflight for the SAM 3 grounding service: checks Hugging Face access and
downloads the gated weights once the license has been accepted.

The SAM 3 repos are gated, so this fails with a clear message until access is
approved at https://huggingface.co/facebook/sam3

Runs in perception_env (Python 3.10), not the robot environment:
    perception_env/bin/python scripts/fetch_sam3.py
"""

import argparse
import os
import shutil
import sys

REPOS = {
    "sam3": ("facebook/sam3", "sam3.pt"),
    "sam3.1": ("facebook/sam3.1", "sam3.1_multiplex.pt"),
}

WEIGHTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=sorted(REPOS), default="sam3")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download, whoami
    from huggingface_hub.errors import GatedRepoError

    try:
        me = whoami()
    except Exception as e:
        print(f"Not authenticated with Hugging Face: {e}")
        print("Run:  perception_env/bin/hf auth login")
        return 1

    print(f"Authenticated as {me.get('name')!r}")

    repo, filename = REPOS[args.version]
    print(f"Fetching {filename} from {repo}...")

    try:
        cached = hf_hub_download(repo, filename)
    except GatedRepoError:
        print(f"\nAccess to {repo} is still gated for this account.")
        print(f"Accept the license at https://huggingface.co/{repo} and re-run.")
        return 1
    except Exception as e:
        print(f"\nDownload failed: {type(e).__name__}: {e}")
        return 1

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    dest = os.path.join(WEIGHTS_DIR, filename)
    expected = os.path.getsize(cached)

    # Copy via a temp file and rename, and compare sizes rather than merely
    # testing existence. A plain "skip if it exists" check will happily accept a
    # half-written file left behind by an interrupted or concurrent run, which
    # for a 3.5 GB checkpoint surfaces much later as a confusing load error.
    if not os.path.exists(dest) or os.path.getsize(dest) != expected:
        tmp = f"{dest}.partial"
        shutil.copy(cached, tmp)
        os.replace(tmp, dest)

    actual = os.path.getsize(dest)
    if actual != expected:
        print(f"\nSize mismatch: {dest} is {actual} bytes, expected {expected}.")
        print("Delete it and re-run.")
        return 1

    print(f"\nReady: {dest} ({actual / 1e9:.2f} GB)")
    print("\nSwitch the grounding service to SAM 3 with:")
    print(f"  perception_env/bin/python perception_service/grounding_service.py "
          f"--backend sam3 --model {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
