"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.db import delete_person_by_name, get_all_people, init_db
from app.schemas import (
    HealthResponse,
    IdentifyFaceResponse,
    PeopleListResponse,
    PersonSummary,
    RecognizeResponse,
    RegisterResponse,
)
from app.services.embedding_service import identify_face, register_person
from app.services.recognition_service import recognize_faces


logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    logger.info("database initialized")
    yield


app = FastAPI(
    title="Face Recognition MVP",
    version="0.1.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def frontend() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse()


@app.post("/register", response_model=RegisterResponse)
async def register(name: str = Form(...), files: list[UploadFile] = File(...)) -> RegisterResponse:
    payload = register_person(name, files)
    return RegisterResponse(**payload)


@app.get("/people", response_model=PeopleListResponse)
async def list_people() -> PeopleListResponse:
    people = [
        PersonSummary(
            name=person.name,
            images_used=person.images_used,
            updated_at=person.updated_at,
        )
        for person in get_all_people()
    ]
    return PeopleListResponse(people=people)


@app.post("/recognize", response_model=RecognizeResponse)
async def recognize(file: UploadFile = File(...)) -> RecognizeResponse:
    matches = recognize_faces(file)
    return RecognizeResponse(matches=matches)


@app.post("/identify-face", response_model=IdentifyFaceResponse)
async def identify(
    name: str = Form(...),
    file: UploadFile = File(...),
    confidence: float | None = Form(default=None),
) -> IdentifyFaceResponse:
    payload = identify_face(name, file, confidence)
    return IdentifyFaceResponse(**payload)


@app.delete("/people/{name}")
async def delete_person(name: str) -> dict[str, str]:
    if not delete_person_by_name(name):
        raise HTTPException(status_code=404, detail="person not found")
    return {"message": "person deleted successfully"}
