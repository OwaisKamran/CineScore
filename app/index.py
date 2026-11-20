"""CineScore Flask app.

Local:   python app/index.py            -> http://localhost:5000
Vercel:  deploy the app/ folder (see app/vercel.json); Vercel serves the `app` object.

Endpoints
  GET  /                 HTML front end
  GET  /api/options      choices for the form (genres, languages, countries, ...)
  GET  /api/search       ?role=director|writer|cast|franchise&q=nol  -> name matches
  POST /predict          JSON film description -> predicted IMDb rating + 80% interval
  GET  /health           liveness check
"""
import os

from flask import Flask, jsonify, render_template, request

try:
    from predictor import Predictor, ValidationError
except ImportError:  # imported as a package (python -m app.index)
    from .predictor import Predictor, ValidationError

app = Flask(__name__)
predictor = Predictor()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/options")
def options():
    return jsonify(predictor.options())


@app.get("/api/search")
def search():
    try:
        return jsonify(predictor.search(request.args.get("role", ""), request.args.get("q", "")))
    except ValidationError as e:
        return jsonify(error=str(e)), 400


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify(error="Send the film details as a JSON body."), 400
    try:
        return jsonify(predictor.predict(payload))
    except ValidationError as e:
        return jsonify(error=str(e)), 400
    except Exception:  # never leak stack traces to the browser
        app.logger.exception("prediction failed")
        return jsonify(error="Something went wrong while predicting. Please try again."), 500


@app.get("/health")
def health():
    return jsonify(status="ok", model=predictor.meta["model"])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
