# Face Recognition MVP

This project is a minimal end-to-end face recognition backend. It lets you register people from one or more uploaded images, stores only averaged face embeddings plus metadata in SQLite, and recognizes faces in a new uploaded image with threshold-based unknown handling.

It now also includes a minimal browser frontend for uploading an image, reviewing confidence scores, and labeling unknown faces so those samples are stored for future recognition.

## Tech stack

- Python 3.11+
- FastAPI
- Uvicorn
- SQLite
- OpenCV
- facenet-pytorch (`MTCNN` + `InceptionResnetV1`)
- NumPy
- Pydantic

## Project structure

```text
face-recognition-mvp/
├── app/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run the server

```bash
uvicorn app.main:app --reload
```

The SQLite database is created automatically at `./face_recognition.db` on first startup.

## API endpoints

### Health check

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

### Register a person

```bash
curl -X POST http://127.0.0.1:8000/register \
  -F "name=Arihant" \
  -F "files=@data/arihant_selfie.jpg" \
  -F "files=@data/arihant_mirror.jpg"
```

Example response:

```json
{
  "name": "Arihant",
  "images_received": 2,
  "images_used": 2,
  "message": "person registered successfully"
}
```

### List people

```bash
curl http://127.0.0.1:8000/people
```

### Recognize faces

```bash
curl -X POST http://127.0.0.1:8000/recognize \
  -F "file=@data/group_indoor.jpg"
```

### Identify an unknown face crop

This endpoint is used by the frontend after you type a name for an unknown face. It stores the labeled face sample in SQLite and refreshes that person's averaged embedding.

```bash
curl -X POST http://127.0.0.1:8000/identify-face \
  -F "name=Arihant" \
  -F "confidence=0.62" \
  -F "file=@data/arihant_selfie.jpg"
```

Harder low-light test:

```bash
curl -X POST http://127.0.0.1:8000/recognize \
  -F "file=@data/group_night.jpg"
```

Example response:

```json
{
  "matches": [
    {
      "box": [120, 80, 260, 240],
      "identity": "Arihant",
      "confidence": 0.81,
      "is_known": true
    }
  ]
}
```

### Optional delete endpoint

```bash
curl -X DELETE http://127.0.0.1:8000/people/Arihant
```

## Configuration

Environment variables:

- `DATABASE_URL`: external Postgres connection string for deployed environments
- `DATABASE_PATH`: optional SQLite file path
- `SIMILARITY_THRESHOLD`: cosine similarity threshold, default `0.70`
- `MAX_UPLOAD_SIZE_BYTES`: maximum accepted upload size, default `5242880`
- `CORS_ALLOWED_ORIGINS`: comma-separated allowed frontend origins, default `*`
- `FACE_DETECTION_CONFIDENCE_THRESHOLD`: detector probability threshold
- `MIN_FACE_BOX_SIZE`: minimum face box size in pixels
- `SECONDARY_DETECTION_SCALE`: second-pass upscale factor for smaller faces
- `DETECTION_MERGE_IOU_THRESHOLD`: duplicate merge threshold for multi-scale detection

## Deployment shape

This project is now set up for two deployment modes:

1. Local demo: FastAPI + SQLite
2. Split deployment: frontend on Vercel, backend on a Python host with Postgres

### Recommended split deployment

- Frontend: Vercel
- Backend: Render, Railway, Fly.io, or another Python host
- Database: Postgres via Neon, Supabase, or another provider

Deployment config files included:

- [render.yaml](/Users/arihantbirani/Documents/faceRecognition/render.yaml) for a Render backend
- [vercel.json](/Users/arihantbirani/Documents/faceRecognition/vercel.json) for a frontend-only Vercel deployment from `app/static`

### Why not Vercel for the backend as-is?

- Vercel does not support durable local SQLite storage in serverless functions.
- This app depends on heavier Python ML libraries, which are a poor fit for a serverless cold-start path.
- Vercel function request limits are not ideal for repeated image uploads.

### Backend deployment notes

- Set `DATABASE_URL` to your Postgres connection string.
- Set `CORS_ALLOWED_ORIGINS` to your Vercel frontend origin, for example:

```bash
CORS_ALLOWED_ORIGINS=https://your-app.vercel.app
```

- Start the backend with:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

On Render, the included `render.yaml` already sets the service shape. You only need to provide:

- `DATABASE_URL`
- `CORS_ALLOWED_ORIGINS`

### Frontend deployment notes

The browser client now supports a separate API base URL. If the frontend is not served by the same backend origin, set one of the following in your deployed HTML:

- `window.__FACE_API_BASE_URL__`
- the `<meta name="face-api-base-url" content="...">` tag in `index.html`

Example:

```html
<meta name="face-api-base-url" content="https://your-backend.example.com">
```

For a frontend-only Vercel deployment from this repo:

1. Import the GitHub repository into Vercel.
2. Let Vercel detect the included [vercel.json](/Users/arihantbirani/Documents/faceRecognition/vercel.json).
3. Vercel will deploy `app/static` as the site output.
4. Update the `face-api-base-url` meta tag in [app/static/index.html](/Users/arihantbirani/Documents/faceRecognition/app/static/index.html) to your backend URL before deploying.

Example backend URL:

```html
<meta name="face-api-base-url" content="https://face-recognition-api.onrender.com">
```

## Development and tests

Run the unit tests:

```bash
pytest
```

## Known limitations

- CPU-only inference by default, so recognition can be slow on larger images.
- The MVP stores one averaged embedding per person, which is simple but less flexible than multiple embeddings.
- Registration uses the largest face in each image, which assumes the primary subject is the intended person.
- SQLite JSON storage is convenient for an MVP but not ideal for larger-scale retrieval.
- Manual identification depends on a clean enough face crop from the browser upload.

## Future improvements

- Add GPU support and configurable model device selection.
- Store multiple embeddings per person and aggregate during matching.
- Add request logging, metrics, and more complete integration tests.
- Add image annotation output for debugging or demos.
- Move to a vector index when the number of registered people grows.
