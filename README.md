GameCam Search
===============

Quick start
-----------

1. Create a virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\Activate.ps1 (PowerShell)
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and set `YOLO_LABELS` (comma-separated):

```text
copy .env.example .env    # PowerShell: Copy-Item .env.example .env
# then edit .env and set YOLO_LABELS, e.g. YOLO_LABELS=person,car
```

3. Run the search over a directory of images:

```bash
python search.py /path/to/images
```

Notes
-----
- Configuration is loaded from the environment via `python-dotenv`.
- `YOLO_MODEL` defaults to `yolov8n.pt` but you can point it to other weights.
- The script prints matching image paths and the detected labels.

Files
-----
- `search.py`: main script that iterates images and runs YOLO
- `requirements.txt`: Python dependencies
- `.env.example`: example environment variables

