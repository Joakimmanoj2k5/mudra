import uvicorn
import os


if __name__ == "__main__":
    host = os.getenv("MUDRA_API_HOST", "127.0.0.1")
    port = int(os.getenv("MUDRA_API_PORT", "8000"))
    uvicorn.run("backend.api:app", host=host, port=port, reload=False)
