"""Minimal external service the agent can query.

A toy "pet shelter" API. Keep it simple — this is starter scaffolding meant to
show the pattern (an external HTTP service the agent reaches over the docker
network), not a full simulation. Add endpoints as the benchmark grows.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Pet Shelter API")

# In-memory seed data. Swap for a real DB (see surgery_bench) when you need
# persistence across requests/restarts.
PETS = [
    {"id": 1, "name": "Biscuit", "species": "dog", "age": 3, "adopted": False},
    {"id": 2, "name": "Mochi", "species": "cat", "age": 1, "adopted": False},
    {"id": 3, "name": "Pickle", "species": "rabbit", "age": 2, "adopted": True},
]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/pets")
def list_pets(available_only: bool = False):
    if available_only:
        return [p for p in PETS if not p["adopted"]]
    return PETS


@app.get("/pets/{pet_id}")
def get_pet(pet_id: int):
    pet = next((p for p in PETS if p["id"] == pet_id), None)
    if pet is None:
        raise HTTPException(status_code=404, detail="Pet not found")
    return pet


class AdoptBody(BaseModel):
    adopter_email: str


@app.post("/pets/{pet_id}/adopt")
def adopt_pet(pet_id: int, body: AdoptBody):
    pet = next((p for p in PETS if p["id"] == pet_id), None)
    if pet is None:
        raise HTTPException(status_code=404, detail="Pet not found")
    if pet["adopted"]:
        raise HTTPException(status_code=409, detail="Pet already adopted")
    pet["adopted"] = True
    pet["adopter_email"] = body.adopter_email
    return {"ok": True, "pet": pet}
