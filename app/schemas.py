"""Pydantic API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class RegisterResponse(BaseModel):
    name: str
    images_received: int
    images_used: int
    message: str


class IdentifyFaceResponse(BaseModel):
    name: str
    images_used: int
    message: str


class PersonSummary(BaseModel):
    name: str
    images_used: int
    updated_at: str


class PeopleListResponse(BaseModel):
    people: list[PersonSummary]


class FaceMatchResponse(BaseModel):
    box: list[int] = Field(min_length=4, max_length=4)
    identity: str
    confidence: float
    is_known: bool


class RecognizeResponse(BaseModel):
    matches: list[FaceMatchResponse]


class ErrorResponse(BaseModel):
    detail: str
