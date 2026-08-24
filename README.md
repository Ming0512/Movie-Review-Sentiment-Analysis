# IMDB Sentiment Analyzer

pip install -r requirements.txt
python main.py        # runs all 4 pipeline stages
python app.py          # http://localhost:5000


A modular, MLOps-style pipeline for a SimpleRNN sentiment classifier trained
on the Keras IMDB movie review dataset, with a Flask web UI for interactive
predictions.

![CI/CD](https://github.com/YOUR_USERNAME/imdb-sentiment-rnn/actions/workflows/main.yaml/badge.svg)

## Project Structure

```
imdb-sentiment-rnn/
├── .github/workflows/
│   └── main.yaml                    # CI/CD: test → build & push to ECR → deploy
├── config/
│   └── config.yaml                   # WHERE artifacts live (paths only)
├── research/
│   └── trials.ipynb                  # Experimentation notebook
├── src/
│   └── sentimentAnalyzer/
│       ├── components/                # The actual ML work
│       │   ├── data_ingestion.py
│       │   ├── data_transformation.py
│       │   ├── model_trainer.py
│       │   └── model_evaluation.py
│       ├── config/
│       │   └── configuration.py       # Reads config.yaml + params.yaml -> entities
│       ├── constants/                 # Paths to config.yaml / params.yaml
│       ├── entity/                    # Dataclasses describing each stage's config
│       ├── logging/                   # Centralized logger
│       ├── pipeline/                  # Orchestrates components into stages
│       │   ├── stage_01_data_ingestion.py
│       │   ├── stage_02_data_transformation.py
│       │   ├── stage_03_model_trainer.py
│       │   ├── stage_04_model_evaluation.py
│       │   └── prediction.py          # Used by app.py at inference time
│       ├── utils/
│       │   └── common.py              # read_yaml, create_directories, save_json...
│       └── __init__.py
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
├── app.py                            # Flask app (UI + /train + /predict)
├── index.html                        # Web UI template (loaded from root)
├── main.py                           # Runs all training stages in sequence
├── params.yaml                       # WHAT the model trains with (hyperparameters)
└── requirements.txt
```

**The separation that matters:** `config/config.yaml` controls *where things
are stored* (artifact paths); `params.yaml` controls *what the model trains
with* (hyperparameters). Changing a learning setting means editing
`params.yaml`, not touching code.

## How the Pipeline Flows

```
main.py
  │
  ├─▶ stage_01_data_ingestion    → downloads IMDB data, saves artifacts/data_ingestion/imdb_raw.npz
  ├─▶ stage_02_data_transformation → pads, shuffles, saves artifacts/data_transformation/{train,test}.npz
  ├─▶ stage_03_model_trainer      → trains SimpleRNN, saves artifacts/model_trainer/best_imdb_model.keras
  └─▶ stage_04_model_evaluation   → evaluates on test set, saves artifacts/model_evaluation/metrics.json
```

Each stage is a thin `pipeline/stage_XX_*.py` wrapper around a
`components/*.py` class, so you can run any single stage in isolation:

```bash
python -m src.sentimentAnalyzer.pipeline.stage_01_data_ingestion
```

## Quickstart

**1. Install dependencies**

```bash
git clone https://github.com/YOUR_USERNAME/imdb-sentiment-rnn.git
cd imdb-sentiment-rnn
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Run the full training pipeline**

```bash
python main.py
```

This runs all four stages in order and populates `artifacts/`. Training
takes a few minutes on CPU.

**3. Launch the web app**

```bash
python app.py
```

Open **http://localhost:5000**, paste a review into the textbox, and click
**Check Sentiment**. You can also trigger training from the browser by
visiting **http://localhost:5000/train**.

## Running with Docker

```bash
# Train first, so artifacts/ exists on the host
python main.py

docker build -t imdb-sentiment-rnn .
docker run -p 5000:5000 -v $(pwd)/artifacts:/app/artifacts imdb-sentiment-rnn
```

> Artifacts are mounted as a volume rather than baked into the image, so you
> can retrain on the host and refresh the container without rebuilding it.

## CI/CD

`.github/workflows/main.yaml` runs three jobs on every push to `main`:

1. **Continuous Integration** — installs dependencies, byte-compiles the
   codebase.
2. **Continuous Delivery** — builds the Docker image and pushes it to Amazon
   ECR.
3. **Continuous Deployment** — (runs on a self-hosted runner) pulls the
   latest image and restarts the container.

**Required repository secrets:**

| Secret | Purpose |
|---|---|
| `AWS_ACCESS_KEY_ID` | AWS credentials |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials |
| `AWS_REGION` | e.g. `us-east-1` |
| `ECR_REPOSITORY_NAME` | Your ECR repo name |
| `AWS_ECR_LOGIN_URI` | e.g. `123456789012.dkr.ecr.us-east-1.amazonaws.com` |

The deployment job needs a self-hosted runner (Settings → Actions →
Runners) registered on the target machine, with Docker installed.

## Notes on Gotchas This Project Handles

1. **Keras's IMDB dataset is sorted by label** (negatives first, then
   positives). `model.fit(..., validation_split=...)` slices the *last* N%
   of the array *without shuffling first*, producing a validation split
   that's almost entirely one class if the data isn't shuffled beforehand.
   `data_transformation.py` shuffles training data (fixed seed) before any
   split happens.

2. **Padding direction matters for a SimpleRNN.** Its final hidden state is
   most influenced by whatever it processed last, so `padding="pre"` (real
   content ending at `max_length`, not buried under trailing padding) works
   far better than `padding="post"` for short inputs. Training and
   inference padding are always read from the same `params.yaml` /
   `model_config.json` source, so they can't silently drift apart.

## Model Architecture

```
Input(shape=(300,))
Embedding(vocab_size=10000, output_dim=32)
SimpleRNN(32, dropout=0.3, recurrent_dropout=0.3)
Dense(1, activation="sigmoid")
```

Trained with `binary_crossentropy` loss and the Adam optimizer, monitoring
`val_accuracy` for both checkpointing and early stopping.

## License

MIT — see [LICENSE](LICENSE).
