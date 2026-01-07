# Corrupted MNIST

Training and evaluating convolutional models on a deliberately corrupted MNIST dataset to explore robust MLOps practices: reproducible data preparation, scripted training/evaluation, automated reporting, and visualization of learned features.

## Highlights
- Typer CLIs for preprocessing, training, evaluation, and embedding visualization with automatic device selection (CUDA, MPS, or CPU).
- Invoke tasks (`tasks.py`) wrapping frequent workflows (`preprocess-data`, `train`, `test`, `docker-build`, documentation builds).
- Data versioning ready: `data/raw/` contains the supplied `.pt` shards, while `data/processed/` stores normalized tensors consumed by the model.
- Model artifacts (`models/`) and figures (`reports/figures/`) are emitted automatically to keep outputs reproducible and centrally located.
- FastAPI + Uvicorn dependencies included for serving predictions once `src/corrupted_mnist/api.py` is completed.

## Quick start
1. Install [uv](https://github.com/astral-sh/uv) (or use any Python 3.12+ environment manager).
2. Create and activate a virtual environment:
	```bash
	uv venv
	source .venv/bin/activate
	```
3. Install the project dependencies (including optional dev tools):
	```bash
	uv sync --all-extras
	```
4. Confirm everything is wired up:
	```bash
	uv run python -m corrupted_mnist.train --epochs 1 --batch-size 64
	```

> Prefer Invoke? Run `uv run invoke --list` to discover the canned workflows.

## Data workflow
- Raw tensors live in `data/raw`, split across `train_images_*.pt` and `train_target_*.pt`, with monolithic test files.
- Processed tensors are normalized, channelized, and written to `data/processed` by `corrupted_mnist.data`.
- Preprocess via Typer or Invoke:
  ```bash
  uv run python -m corrupted_mnist.data data/raw data/processed
  # or
  uv run invoke preprocess-data
  ```

## Training
`src/corrupted_mnist/train.py` defines a minimal CNN trained with Adam and cross-entropy.
```bash
uv run python -m corrupted_mnist.train --lr 5e-4 --batch-size 64 --epochs 15
# or equivalently
uv run invoke train
```
Artifacts:
- `models/model.pth`: serialized `state_dict`.
- `reports/figures/training_statistics.png`: loss/accuracy curves across iterations.

## Evaluation
`src/corrupted_mnist/evaluate.py` loads the saved checkpoint and reports accuracy on the processed test split.
```bash
uv run python -m corrupted_mnist.evaluate --model-checkpoint models/model.pth --batch-size 128
```

## Feature visualization
Use t-SNE (with optional PCA pre-reduction) to inspect the penultimate-layer embeddings of a trained model.
```bash
uv run python -m corrupted_mnist.visualize --model-checkpoint models/model.pth --split test --max-points 4000
```
Outputs land in `reports/figures/<figure_name>`.

## API stub
FastAPI and Uvicorn are available for serving predictions once `src/corrupted_mnist/api.py` is implemented. A typical launch command will look like:
```bash
uv run uvicorn corrupted_mnist.api:app --reload --port 8000
```
Use this section as a reminder to expose `MyModel` through a REST interface when needed.

## Testing and quality
Run the automated test suite plus coverage reporting:
```bash
uv run invoke test
# internally executes
# uv run coverage run -m pytest tests/
# uv run coverage report -m -i
```
Additional tooling:
```bash
uv run ruff check src tests
uv run pre-commit run --all-files
```

## Documentation
MkDocs Material powers the docs site located in `docs/source`. Use the dedicated config file:
```bash
uv run invoke build-docs
uv run invoke serve-docs  # auto-reloads while editing docs
```

## Docker images
Build ready-to-train and API-serving images:
```bash
uv run invoke docker-build --progress=plain
```
The Dockerfiles live under `dockerfiles/` and assume the same entrypoints as the Typer CLIs.

## Project layout
```text
├── configs/                 # Hydra/experiment configs (extend as needed)
├── data/
│   ├── processed/           # Normalized tensors consumed by loaders
│   └── raw/                 # Provided corrupted MNIST shards (.pt)
├── dockerfiles/
│   ├── api.dockerfile
│   └── train.dockerfile
├── docs/
│   ├── mkdocs.yaml
│   └── source/
├── models/                  # Saved checkpoints
├── reports/
│   └── figures/             # Training stats, t-SNE plots, etc.
├── src/
│   └── corrupted_mnist/
│       ├── __init__.py
│       ├── api.py           # FastAPI skeleton
│       ├── data.py          # Preprocessing + dataset helpers
│       ├── evaluate.py      # Test-set evaluation CLI
│       ├── model.py         # CNN definition
│       ├── train.py         # Training CLI
│       └── visualize.py     # Embedding visualization CLI
├── tests/
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── tasks.py                 # Invoke automation hooks
├── pyproject.toml           # Project + dependency metadata
└── README.md
```

## Next steps
- Finish `api.py` with prediction endpoints and wire up FastAPI tests.
- Add experiment tracking (e.g., MLflow or Weights & Biases) for hyperparameter sweeps.
- Extend `configs/` with structured configs and surface them via Typer to keep experiments reproducible.
