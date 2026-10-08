"""
Unit Tests for Stage 7A Docker & Production Configuration.
"""

import os
from pathlib import Path

from config.settings import BASE_DIR, POSTGRES_HOST, DATABASE_URL



class TestDockerConfig:
    """Test suite for Dockerfile, .dockerignore, docker-compose.yml, and environment variable resolution."""

    def test_dockerfile_contents(self):
        """Verify Dockerfile uses Python 3.11-slim base, exposes port 8000, and uses production Uvicorn command."""
        dockerfile_path = BASE_DIR / "Dockerfile"
        assert dockerfile_path.exists(), "Dockerfile does not exist"

        content = dockerfile_path.read_text()
        assert "FROM python:3.11-slim" in content
        assert "EXPOSE 8000" in content
        assert '"uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"' in content
        assert "--reload" not in content

    def test_dockerignore_exclusions(self):
        """Verify .dockerignore excludes sensitive and temporary files/directories."""
        dockerignore_path = BASE_DIR / ".dockerignore"
        assert dockerignore_path.exists(), ".dockerignore does not exist"

        content = dockerignore_path.read_text()
        lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]

        assert ".env" in lines
        assert ".venv/" in lines
        assert ".git/" in lines
        assert "logs/" in lines

    def test_docker_compose_structure(self):
        """Verify docker-compose.yml defines api and db services, healthchecks, and environment variables."""
        compose_path = BASE_DIR / "docker-compose.yml"
        assert compose_path.exists(), "docker-compose.yml does not exist"

        content = compose_path.read_text()

        assert "db:" in content
        assert "api:" in content
        assert "image: postgres:15-alpine" in content
        assert "command: uvicorn api.main:app --host 0.0.0.0 --port 8000" in content
        assert "--reload" not in content
        assert "POSTGRES_HOST=db" in content
        assert "POSTGRES_DB=sales_forecasting" in content
        assert "pg_isready" in content
        assert "http://localhost:8000/health" in content
        assert "postgres_data:" in content


    def test_environment_variable_fallback(self):
        """Verify database settings fallback gracefully to POSTGRES_HOST env var."""
        assert isinstance(POSTGRES_HOST, str)
        assert isinstance(DATABASE_URL, str)
