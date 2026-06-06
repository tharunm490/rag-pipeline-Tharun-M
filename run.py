import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),  # Render sets PORT automatically
        reload=False,                         # Never reload in production
        workers=1,                            # Free tier = 1 worker only
    )