"""Publish CineScore to Hugging Face: a model repo (pipeline + model card) and a static Space that
runs the Streamlit app in the browser with stlite (Streamlit on Pyodide). Static Spaces are free;
Docker/Gradio Spaces on the free CPU tier need a PRO account.

Log in once first (the token is stored by huggingface_hub, never in this repo):
    .venv\\Scripts\\hf auth login

Then:
    python scripts/06_publish_huggingface.py                       # model repo + Space, logged-in account
    python scripts/06_publish_huggingface.py --only model          # just the model repo
    python scripts/06_publish_huggingface.py --stage-only build/hf # build the Space folder, no upload
    (serve a staged folder locally with:  python -m http.server -d build/hf 8765)

Creates or updates   https://huggingface.co/<user>/cinescore          (model)
                     https://huggingface.co/spaces/<user>/cinescore   (Space)
"""
import argparse
import json
import shutil
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF = ROOT / "huggingface"
NAME = "cinescore"
# Everything the Streamlit app needs at run time; nothing else from the repo goes into the Space.
SPACE_FILES = ["streamlit_app/streamlit_app.py", "streamlit_app/streamlit.css", "app/predictor.py",
               "app/static/tokens.css", "app/data/lookups.json.gz", "app/data/country_names.json",
               "app/data/rating_stats.json", "app/data/demo_2026.json", "app/model/cinescore_pipeline.joblib", "app/model/metadata.json"]


def fill(text: str, user: str) -> str:
    return text.replace("{user}", user)


def streamlit_config() -> dict:
    """.streamlit/config.toml as stlite's streamlitConfig ({"theme.base": "light", ...})."""
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    out = {f"{section}.{k}": v for section, values in cfg.items() for k, v in values.items()
           if section in ("theme", "client")}
    # "Name:https://..., serif" -> "Name": web-font URLs in the theme crash stlite 1.12's page-head manager,
    # and stlite treats "Name, serif" as one font name. The fonts themselves are loaded by tokens.css.
    for k in ("theme.font", "theme.headingFont"):
        if k in out:
            out[k] = out[k].split(":", 1)[0].split(",", 1)[0].strip()
    return out


def stage_space(dest: Path, user: str) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    page = (HF / "space" / "index.html").read_text(encoding="utf-8")
    page = page.replace("/*FILES*/[]", json.dumps(SPACE_FILES)).replace("/*CONFIG*/{}", json.dumps(streamlit_config()))
    (dest / "index.html").write_text(page, encoding="utf-8")
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
    ap.add_argument("--only", choices=["model", "space"], help="publish just one of the two repos")
    ap.add_argument("--stage-only", type=Path, help="only build the Space folder here, do not upload")
    args = ap.parse_args()

    if args.stage_only:
        print("staged:", stage_space(args.stage_only, args.user or "USER"))
        return

    from huggingface_hub import HfApi
    api = HfApi()
    user = args.user or api.whoami()["name"]
    repo_id = f"{user}/{NAME}"

    with tempfile.TemporaryDirectory() as tmp:
        if args.only in (None, "model"):
            api.create_repo(repo_id, repo_type="model", exist_ok=True)
            api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=stage_model(Path(tmp) / "model", user),
                              commit_message="CineScore pipeline (scikit-learn 1.8.0) and model card")
            print(f"model: https://huggingface.co/{repo_id}")
        if args.only in (None, "space"):
            api.create_repo(repo_id, repo_type="space", space_sdk="static", exist_ok=True)
            api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=stage_space(Path(tmp) / "space", user),
                              commit_message="CineScore Streamlit app in the browser (stlite)",
                              delete_patterns=["app/**", "streamlit_app/**"])
            print(f"space: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
