from tortoise import Tortoise
from tortoise.backends.sqlite.client import SqliteClient
import os
from dotenv import load_dotenv

# Allow aiosqlite worker-thread connections (especially sqlite://:memory:) to be
# shared safely between pytest's async loop and Starlette's TestClient portal thread.
SqliteClient._check_loop = lambda self: True

load_dotenv(override=False)

if os.getenv("TESTING") == "1":
    DATABASE_URL = os.getenv("TEST_DATABASE_URL", "sqlite://:memory:")
else:
    DATABASE_URL = os.getenv("DATABASE_URL", "postgres://user:password@localhost:5432/liberia")

TORTOISE_ORM = {
    "connections": {"default": DATABASE_URL},
    "apps": {
        "models": {
            "models": ["app.models", "aerich.models"],
            "default_connection": "default",
        },
    },
}
