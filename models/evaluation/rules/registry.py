"""規則註冊表。"""

from __future__ import annotations

from typing import Iterable, Optional

from models.evaluation.rules.base import BettingRule
from models.evaluation.rules.definitions import (
    DEFAULT_REPORT_RULE_IDS,
    all_builtin_rules,
)


class RuleRegistry:
    
    def __init__(self) -> None:
        self._rules: dict[str, BettingRule] = {}

    def register(self, rule: BettingRule) -> None:
        if not rule.rule_id:
            raise ValueError(f"rule_id 不可為空: {rule}")
        self._rules[rule.rule_id] = rule

    def register_many(self, rules: Iterable[BettingRule]) -> None:
        for r in rules:
            self.register(r)

    def get(self, rule_id: str) -> BettingRule:
        if rule_id not in self._rules:
            raise KeyError(f"未註冊規則: {rule_id}")
        return self._rules[rule_id]

    def has(self, rule_id: str) -> bool:
        return rule_id in self._rules

    def list(
        self, ids: Optional[list[str]] = None
    ) -> list[BettingRule]:
        if ids is None:
            return list(self._rules.values())
        out = []
        for i in ids:
            if i in self._rules:
                out.append(self._rules[i])
        return out

    def ids(self) -> list[str]:
        return list(self._rules.keys())

    def describe(self) -> list[dict]:
        return [r.to_dict() for r in self._rules.values()]


def default_registry() -> RuleRegistry:
    reg = RuleRegistry()
    reg.register_many(all_builtin_rules())
    return reg


def default_report_ids() -> list[str]:
    return list(DEFAULT_REPORT_RULE_IDS)