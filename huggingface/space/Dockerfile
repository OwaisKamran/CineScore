# CineScore Hugging Face Space: the Streamlit app (streamlit_app/) on port 7860.
# Hugging Face retired the built-in Streamlit SDK for new Spaces (2025-04-30); Docker is the
# recommended way to run Streamlit there.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user streamlit_app/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# .streamlit/config.toml (theme), app/ (predictor, model, lookups, tokens.css), streamlit_app/
COPY --chown=user . .

EXPOSE 7860
CMD ["streamlit", "run", "streamlit_app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
