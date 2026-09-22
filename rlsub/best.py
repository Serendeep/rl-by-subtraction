"""Brute-forces the environment's optimum so trained policies have something to miss.

True satisfaction depends only on the multiset of items, so enumerating
combinations with replacement covers the whole space.
"""

from __future__ import annotations

from itertools import combinations_with_replacement

from .deli import MAX_ITEMS, VOCAB_SIZE, describe, price, true_satisfaction, verifies


def ranked() -> list[tuple[float, tuple[int, ...]]]:
    scored = [
        (true_satisfaction(combo), combo)
        for n in range(1, MAX_ITEMS + 1)
        for combo in combinations_with_replacement(range(VOCAB_SIZE), n)
    ]
    scored.sort(reverse=True)
    return scored


def main() -> None:
    scored = ranked()
    best_score, best_order = scored[0]
    verifying = next((pair for pair in scored if verifies(pair[1])), None)

    print(f"best possible       {best_score:+.2f}  £{price(best_order):5.2f}  {describe(best_order)}")
    if verifying:
        score, order = verifying
        print(f"best that verifies  {score:+.2f}  £{price(order):5.2f}  {describe(order)}")
    worst_score, worst_order = scored[-1]
    print(f"worst               {worst_score:+.2f}  £{price(worst_order):5.2f}  {describe(worst_order)}")


if __name__ == "__main__":
    main()
