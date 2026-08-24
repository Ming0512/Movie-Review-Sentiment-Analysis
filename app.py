"""Run locally:
    python app.py
Then open http://localhost:5000 in your browser.
"""

import os

from flask import Flask, render_template, request

from src.sentimentAnalyzer.logging import logger

# index.html lives at the project root (not in a templates/ folder),
# so point Flask's template_folder at the current directory.
app = Flask(__name__, template_folder=".", static_folder="static")

_prediction_pipeline = None


def get_prediction_pipeline():
    """Lazy-load the PredictionPipeline (and the trained model) once,
    on first request, and cache it."""
    global _prediction_pipeline
    if _prediction_pipeline is None:
        from src.sentimentAnalyzer.pipeline.prediction import PredictionPipeline
        _prediction_pipeline = PredictionPipeline()
    return _prediction_pipeline


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", review=None, label=None, score=None, error=None)


@app.route("/predict", methods=["POST"])
def predict():
    review_text = request.form.get("review", "").strip()

    if not review_text:
        return render_template(
            "index.html", review=None, label=None, score=None,
            error="Please enter a review before submitting."
        )

    try:
        pipeline = get_prediction_pipeline()
    except Exception as e:
        logger.exception(e)
        return render_template(
            "index.html", review=review_text, label=None, score=None,
            error=(
                "No trained model found. Train it first by visiting /train "
                "or running: python main.py"
            ),
        )

    label, score = pipeline.predict(review_text)

    return render_template(
        "index.html", review=review_text, label=label, score=f"{score:.4f}", error=None
    )


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
