"""Trainer registry with duplicate-name protection."""

from __future__ import annotations

from src.training.base import ModelTrainer


class TrainerRegistry:
    def __init__(self) -> None:
        self._trainers: dict[str, ModelTrainer] = {}

    def register(self, trainer: ModelTrainer) -> None:
        name = trainer.metadata().name
        if name in self._trainers:
            raise ValueError(f"trainer {name!r} is already registered")
        self._trainers[name] = trainer

    def get(self, name: str) -> ModelTrainer:
        try:
            return self._trainers[name]
        except KeyError as exc:
            raise KeyError(f"unknown trainer {name!r}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._trainers))
