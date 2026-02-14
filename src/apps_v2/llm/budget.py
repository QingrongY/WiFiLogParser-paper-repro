from __future__ import annotations

from dataclasses import dataclass


class BudgetExceeded(RuntimeError):
    """Raised when upstream quota policy blocks an LLM call."""


@dataclass(frozen=True)
class Reservation:
    reserved_session_tokens: int = 0
    reserved_daily_tokens: int = 0


def reserve_tokens_for_call(**_: object) -> Reservation | None:
    return None


def release_reservation(_: Reservation | None) -> None:
    return None


def commit_usage(*, reservation: Reservation, actual_total_tokens: int) -> None:
    _ = reservation
    _ = actual_total_tokens
    return None
