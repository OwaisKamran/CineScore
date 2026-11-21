"""Publish CineScore to Hugging Face: a model repo (pipeline + model card) and a Docker Space
running the Streamlit app (Hugging Face serves Streamlit through the Docker SDK since 2025).

Log in once first (the token is stored by huggingface_hub, never in this repo):
    .venv\\Scripts\\hf auth login

Then:
    python scripts/06_publish_huggingface.py              # uses the logged-in account
    python scripts/06_publish_huggingface.py --stage-only build/hf_space   # just build the Space folder

Creates or updates   https://huggingface.co/<user>/cinescore          (model)
                     https://huggingface.co/spaces/<user>/cinescore   (Space)
"""
import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF = ROOT / "huggingface"
NAME = "cinescore"
# Everything the Streamlit app needs at run time; nothing else from the repo goes into the Space.
SPACE_FILES = [".streamlit/config.toml", "streamlit_app/streamlit_app.py", "streamlit_app/streamlit.css",
               "streamlit_app/requirements.txt", "app/predictor.py", "app/static/tokens.css",
               "app/data/lookups.json.gz", "app/data/country_names.json",
               "app/model/cinescore_pipeline.joblib", "app/model/metadata.json"]


def fill(text: str, user: str) -> str:
    return text.replace("{user}", user)


def stage_space(dest: Path, user: str) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HF / "space" / "Dockerfile", dest / "Dockerfile")
    (dest / "README.md").write_text(fill((HF / "space" / "README.md").read_text(encoding="utf-8"), user),
                                    encoding="utf-8")
    for rel in SPACE_FILES:
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, out)
    return dest


def stage_model(dest: Path, user: str) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for f in ["cinescore_pipeline.joblib", "metadata.json"]:
        shutil.copy2(ROOT / "app" / "model" / f, dest / f)
    (dest / "README.md").write_text(fill((HF / "model_card.md").read_text(encoding="utf-8"), user),
                                    encoding="utf-8")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", help="Hugging Face username (default: the logged-in account)")
    ap.add_argument("--stage-only", type=Path, help="only build the Space folder here, do not upload")
    args = ap.parse_args()

    if args.stage_only:
        print("staged:", stage_space(args.stage_only, args.user or "USER"))
        return

    from huggingface_hub import HfApi
    api = HfApi()
    user = args.user or api.whoami()["name"]
    model_id, space_id = f"{user}/{NAME}", f"{user}/{NAME}"

    with tempfile.TemporaryDirectory() as tmp:
        api.create_repo(model_id, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=model_id, repo_type="model", folder_path=stage_model(Path(tmp) / "model", user),
                          commit_message="CineScore pipeline and model card")
        api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(repo_id=space_id, repo_type="space", folder_path=stage_space(Path(tmp) / "space", user),
                          commit_message="CineScore Streamlit app (Docker)",
                          delete_patterns=["app/**", "streamlit_app/**", ".streamlit/**"])
    print(f"model: https://huggingface.co/{model_id}")
    print(f"space: https://huggingface.co/spaces/{space_id}  (first build takes a few minutes)")


if __name__ == "__main__":
    main()
