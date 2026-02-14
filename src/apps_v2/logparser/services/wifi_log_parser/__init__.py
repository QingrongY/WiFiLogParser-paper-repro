"""WiFiLogParser hybrid engine.

This engine follows the WiFiLogParser paper flow:
- signature micro-grouping (reuse LogExtractor skeleton clustering)
- LLM-induced parser per large micro-group
- coverage-driven consolidation to merge fragmented micro-groups

Implementation note: we preserve LogExtractor's min-cluster-size skip behaviour
(no LLM calls for tiny clusters), but allow tiny clusters to be merged into an
existing parser bucket if they are covered by a validated regex.
"""

from __future__ import annotations

__all__ = ["WiFiLogParserEngine"]

from .engine import WiFiLogParserEngine
