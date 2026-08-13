from __future__ import annotations

from collections import Counter
from typing import Iterable

from src.importers.models import ValidationIssue


def count_by_severity(issues: Iterable[ValidationIssue]) -> dict[str, int]:
    counts = Counter(issue.severity for issue in issues)
    return {"ERROR": counts["ERROR"], "WARNING": counts["WARNING"], "INFO": counts["INFO"]}


def verdict_from_issues(issues: Iterable[ValidationIssue]) -> str:
    return "NOT READY - FIX MAPPING ISSUES FIRST" if any(issue.severity == "ERROR" for issue in issues) else "READY FOR DATABASE INSERT TEST"
