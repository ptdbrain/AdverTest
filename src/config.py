from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_name: str = "AdverTest"
    app_env: Literal["development", "production", "test"] = "development"
    app_port: int = Field(default=8000, ge=1, le=65535)
    app_host: str = "0.0.0.0"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    cors_origins: str = "http://localhost:3000"

    # Test-run defaults (plan §5: cheap by default, opt into expensive scans)
    default_model: str = "blob_detector"
    default_dataset: str = "synthetic_shapes"
    default_sample_limit: int = Field(default=8, ge=1)
    default_severities: str = "1,3,5"
    run_seed: int = 20260730
    #: Calibration for the pre-run cost estimate: seconds per unit of attack cost.
    seconds_per_cost_unit: float = Field(default=0.05, gt=0.0)

    # Human-in-the-loop gate (plan §7): degradation above this needs a Reviewer.
    review_degradation_threshold: float = Field(default=0.30, ge=0.0, le=1.0)

    # Storage (in-memory today; PostgreSQL + MinIO per plan §4)
    database_url: str = "sqlite:///./data/app.db"

    @property
    def severity_list(self) -> list[int]:
        """``default_severities`` parsed into integers."""
        return [int(part) for part in self.default_severities.split(",") if part.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
