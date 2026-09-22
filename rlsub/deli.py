"""Tiny deli environment: one hidden true reward, three partial views of it."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Category(str, Enum):
    BREAD = "bread"
    FILLING = "filling"
    SAUCE = "sauce"
    EXTRA = "extra"


@dataclass(frozen=True)
class Item:
    name: str
    category: Category
    price: float
    tastiness: float


MENU: tuple[Item, ...] = (
    Item("sourdough", Category.BREAD, 1.20, 0.55),
    Item("rye", Category.BREAD, 1.30, 0.35),
    Item("focaccia", Category.BREAD, 1.60, 0.95),
    Item("halloumi", Category.FILLING, 2.40, 1.95),
    Item("turkey", Category.FILLING, 2.60, 1.55),
    Item("falafel", Category.FILLING, 1.90, 1.35),
    Item("harissa", Category.SAUCE, 0.40, 0.15),
    Item("aioli", Category.SAUCE, 0.40, -0.05),
    Item("pickles", Category.EXTRA, 0.50, -0.25),
    Item("rocket", Category.EXTRA, 0.45, -0.45),
    Item("tomato", Category.EXTRA, 0.35, -0.40),
    Item("chilli", Category.EXTRA, 0.30, -0.55),
)

VOCAB_SIZE = len(MENU)
MAX_ITEMS = 12  # 3x COMFORTABLE_ITEMS, so the cap never shapes the result

# The trap. Below this, more items really is better. Past it, crowding wins.
COMFORTABLE_ITEMS = 4
CROWDING_WEIGHT = 0.22

STRUCTURE_BONUS = 0.50  # exactly one bread and at least one filling
SAUCE_CLASH_PENALTY = 0.35  # per sauce beyond the first
BUDGET = 6.00
# Per pound over, not flat. A flat penalty is bounded, so a blown budget costs
# nothing extra to blow further.
OVER_BUDGET_RATE = 0.60

# Logistic read of true satisfaction, which gives a ground-truth probability.
ACCEPT_MIDPOINT = 2.20
ACCEPT_SHARPNESS = 2.10


Order = tuple[int, ...]


def item(token: int) -> Item:
    return MENU[token]


def price(order: Order) -> float:
    return sum(MENU[t].price for t in order)


def _count_category(order: Order, category: Category) -> int:
    return sum(1 for t in order if MENU[t].category is category)


def true_satisfaction(order: Order) -> float:
    """Never visible to any training method."""
    if not order:
        return 0.0

    score = sum(MENU[t].tastiness for t in order)
    score += _structure_term(order)
    score -= CROWDING_WEIGHT * max(0, len(order) - COMFORTABLE_ITEMS) ** 2

    score -= OVER_BUDGET_RATE * max(0.0, price(order) - BUDGET)

    return float(score)


def _structure_term(order: Order) -> float:
    term = 0.0
    if _count_category(order, Category.BREAD) == 1 and _count_category(order, Category.FILLING) >= 1:
        term += STRUCTURE_BONUS
    term -= SAUCE_CLASH_PENALTY * max(0, _count_category(order, Category.SAUCE) - 1)
    return term


def accept_probability(order: Order) -> float:
    """Ground truth RLCD calibrates against."""
    logit = ACCEPT_SHARPNESS * (true_satisfaction(order) - ACCEPT_MIDPOINT)
    return float(1.0 / (1.0 + np.exp(-logit)))


def sample_acceptance(order: Order, rng: np.random.Generator) -> int:
    return int(rng.random() < accept_probability(order))


# --- RLVR ------------------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    rule: str


def violations(order: Order) -> tuple[Violation, ...]:
    found: list[Violation] = []
    if not order:
        found.append(Violation("empty order"))
    if _count_category(order, Category.BREAD) != 1:
        found.append(Violation("needs exactly one bread"))
    if _count_category(order, Category.FILLING) < 1:
        found.append(Violation("needs at least one filling"))
    if len(order) > MAX_ITEMS:
        found.append(Violation(f"more than {MAX_ITEMS} items"))
    if price(order) > BUDGET:
        found.append(Violation(f"over budget ({price(order):.2f} > {BUDGET:.2f})"))
    return tuple(found)


def verifies(order: Order) -> bool:
    return not violations(order)


def describe(order: Order) -> str:
    return " + ".join(MENU[t].name for t in order) if order else "(empty)"
