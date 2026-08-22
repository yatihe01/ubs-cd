from fastapi import FastAPI

from routers import adaptive_gateway

app = FastAPI()
app.include_router(adaptive_gateway.router)


@app.get("/")
def health():
    return {"status": "ok"}