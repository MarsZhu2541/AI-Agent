from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"
