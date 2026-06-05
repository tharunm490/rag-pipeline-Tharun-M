import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import uvicorn

if __name__ == "__main__":
    # reload=False in production/Docker, set to True for local dev
    is_dev = os.getenv("ENV", "production") == "development"
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=is_dev)