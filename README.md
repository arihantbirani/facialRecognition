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

Open the frontend at [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Suggested local test image layout

Create a `data/` folder with names like:

```text
data/
├── arihant_selfie.jpg
├── arihant_mirror.jpg
├── group_indoor.jpg
└── group_night.jpg
```

Recommended use:

- Register with `arihant_selfie.jpg`
- Optionally add `arihant_mirror.jpg`
- Recognize against `group_indoor.jpg`
- Use `group_night.jpg` as a harder low-light test

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

- `DATABASE_PATH`: optional SQLite file path
- `SIMILARITY_THRESHOLD`: cosine similarity threshold, default `0.70`
- `MAX_UPLOAD_SIZE_BYTES`: maximum accepted upload size, default `5242880`

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
