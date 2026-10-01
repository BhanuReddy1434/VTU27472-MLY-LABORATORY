# Task 4: Apriori and FP-Growth Association Rules
# Dataset: sample book transactions, following the laboratory manual.

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

transactions = [
    ["Book1", "Book2", "Book3"],
    ["Book2", "Book3", "Book4"],
    ["Book1", "Book3", "Book5"],
    ["Book2", "Book4", "Book5"],
]

encoder = TransactionEncoder()
encoded = encoder.fit(transactions).transform(transactions)
df = pd.DataFrame(encoded, columns=encoder.columns_)

min_support = 0.20
min_confidence = 0.50

print("=== APRIORI ===")
apriori_sets = apriori(df, min_support=min_support, use_colnames=True)
print(apriori_sets)

apriori_rules = association_rules(
    apriori_sets, metric="confidence", min_threshold=min_confidence
)
print("\nApriori Association Rules:")
print(apriori_rules)

print("\n=== FP-GROWTH ===")
fp_sets = fpgrowth(df, min_support=min_support, use_colnames=True)
print(fp_sets)

fp_rules = association_rules(
    fp_sets, metric="confidence", min_threshold=min_confidence
)
print("\nFP-Growth Association Rules:")
print(fp_rules)
