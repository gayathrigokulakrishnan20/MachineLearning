"""Association rules exercise using the Apriori algorithm.

Install the dependency with: pip install apyori

Run this file directly to use the built-in sample baskets. To use a CSV instead,
place retail_dataset.csv beside this file. The CSV should have one transaction
per row and one item per column; blank cells are ignored.
"""

from pathlib import Path
import csv

from apyori import apriori


SAMPLE_TRANSACTIONS = [
    ["bread", "milk"],
    ["bread", "diapers", "beer", "eggs"],
    ["milk", "diapers", "beer", "cola"],
    ["bread", "milk", "diapers", "beer"],
    ["bread", "milk", "diapers", "cola"],
    ["bread", "butter"],
    ["milk", "butter"],
    ["bread", "milk", "eggs"],
    ["diapers", "beer"],
    ["bread", "diapers", "beer"],
    ["milk", "diapers", "cola"],
    ["bread", "milk", "diapers", "beer"],
]


def load_transactions(csv_path: Path) -> list[list[str]]:
    """Read a basket-per-row CSV and drop empty cells and empty rows."""
    transactions = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.reader(file):
            items = [cell.strip() for cell in row if cell.strip()]
            if items:
                transactions.append(items)
    return transactions


def get_transactions() -> list[list[str]]:
    """Load the optional retail CSV, or use the included teaching data."""
    csv_path = Path(__file__).with_name("retail_dataset.csv")
    if csv_path.exists():
        transactions = load_transactions(csv_path)
        print(f"Loaded {len(transactions)} transactions from {csv_path.name}.")
        return transactions

    print("retail_dataset.csv not found; using the built-in sample baskets.")
    return SAMPLE_TRANSACTIONS


def mine_rules(transactions: list[list[str]]) -> list[dict[str, object]]:
    """Mine frequent itemsets and convert Apriori output into readable rules."""
    # Support is the fraction of baskets containing an itemset.
    # Confidence is P(consequent | antecedent); lift > 1 indicates positive association.
    results = apriori(
        transactions,
        min_support=0.15,
        min_confidence=0.30,
        min_lift=1.0,
        min_length=2,
        max_length=3,
    )

    rules = []
    for itemset_result in results:
        for statistic in itemset_result.ordered_statistics:
            if not statistic.items_base or not statistic.items_add:
                continue
            rules.append(
                {
                    "antecedent": sorted(statistic.items_base),
                    "consequent": sorted(statistic.items_add),
                    "support": itemset_result.support,
                    "confidence": statistic.confidence,
                    "lift": statistic.lift,
                }
            )

    return sorted(rules, key=lambda rule: (rule["lift"], rule["confidence"]), reverse=True)


def main() -> None:
    transactions = get_transactions()
    if not transactions:
        raise ValueError("No transactions found. Check that the CSV has basket rows.")

    print(f"Number of transactions: {len(transactions)}")
    print(f"Example transaction: {transactions[0]}")

    rules = mine_rules(transactions)
    print(f"\nAssociation rules found: {len(rules)}")
    if not rules:
        print("No rules met the thresholds. Try lowering min_support or min_confidence.")
    else:
        print("\nTop rules (sorted by lift):")
        for rule in rules[:15]:
            left = ", ".join(rule["antecedent"])
            right = ", ".join(rule["consequent"])
            print(
                f"  {{{left}}} -> {{{right}}} | "
                f"support={rule['support']:.3f}, "
                f"confidence={rule['confidence']:.3f}, "
                f"lift={rule['lift']:.2f}"
            )

    # EXERCISE: change one setting at a time and compare the resulting rules.
    # 1. Raise min_support to 0.25. Which rules disappear?
    # 2. Raise min_confidence to 0.70. How does the rule list change?
    # 3. Find a rule with lift > 1 and explain what its support, confidence,
    #    and lift say about the transactions. Does it imply causation?
    # 4. Add a few transactions to SAMPLE_TRANSACTIONS and observe the changes.


if __name__ == "__main__":
    main()
