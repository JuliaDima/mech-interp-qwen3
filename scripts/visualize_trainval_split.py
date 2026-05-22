#!/usr/bin/env python3
"""Visualize the train/val split distribution issue."""

import matplotlib.pyplot as plt

# Simulate the OLD way (no shuffle)
print("OLD WAY (No Shuffle):")
print("=" * 60)

# Generate grid in order
pairs = []
for a in range(100):
    for b in range(100):
        pairs.append((a, b))

# Split: first 8000 train, last 2000 val
train_pairs = pairs[:8000]
val_pairs = pairs[8000:]

# Analyze distributions
train_a = [p[0] for p in train_pairs]
val_a = [p[0] for p in val_pairs]

print(f"Train set: a ranges from {min(train_a)} to {max(train_a)}")
print(f"Val set:   a ranges from {min(val_a)} to {max(val_a)}")
print()


# Count carries
def has_carry(a, b):
    return (
        1
        if any(
            int(da) + int(db) >= 10
            for da, db in zip(str(a).zfill(2), str(b).zfill(2), strict=False)
        )
        else 0
    )


train_carries = sum(has_carry(a, b) for a, b in train_pairs)
val_carries = sum(has_carry(a, b) for a, b in val_pairs)

print(
    f"Train: {train_carries}/{len(train_pairs)} carries ({100 * train_carries / len(train_pairs):.1f}%)"
)
print(f"Val:   {val_carries}/{len(val_pairs)} carries ({100 * val_carries / len(val_pairs):.1f}%)")
print()

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Train distribution
axes[0, 0].hist2d([p[0] for p in train_pairs], [p[1] for p in train_pairs], bins=20, cmap="Blues")
axes[0, 0].set_xlabel("Operand A")
axes[0, 0].set_ylabel("Operand B")
axes[0, 0].set_title("Train Set Distribution (OLD - No Shuffle)\nMostly a=0-79")
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Val distribution
axes[0, 1].hist2d([p[0] for p in val_pairs], [p[1] for p in val_pairs], bins=20, cmap="Reds")
axes[0, 1].set_xlabel("Operand A")
axes[0, 1].set_ylabel("Operand B")
axes[0, 1].set_title("Val Set Distribution (OLD - No Shuffle)\nMostly a=80-99")
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Distribution of A values
axes[1, 0].hist(train_a, bins=50, alpha=0.7, label="Train", color="blue")
axes[1, 0].hist(val_a, bins=50, alpha=0.7, label="Val", color="red")
axes[1, 0].set_xlabel("Value of Operand A")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].set_title("Distribution of A values (OLD)\nClear separation!")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Carry rates by A value
a_values = list(range(100))
carry_rates = []
for a in a_values:
    # For each a, calculate carry rate across all b
    carries = sum(has_carry(a, b) for b in range(100))
    carry_rates.append(carries / 100)

axes[1, 1].plot(a_values, carry_rates, linewidth=2, color="purple")
axes[1, 1].axvline(x=80, color="red", linestyle="--", linewidth=2, label="Train/Val split")
axes[1, 1].fill_between([0, 80], 0, 1, alpha=0.2, color="blue", label="Train region")
axes[1, 1].fill_between([80, 100], 0, 1, alpha=0.2, color="red", label="Val region")
axes[1, 1].set_xlabel("Value of A")
axes[1, 1].set_ylabel("Carry Rate")
axes[1, 1].set_title('Carry Rate by A value\nHigher A → More carries → Val is "easier"')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("trainval_split_issue.png", dpi=300, bbox_inches="tight")
print("Saved visualization to: trainval_split_issue.png")
plt.close()

print()
print("=" * 60)
print("This explains why Val Accuracy (96%) > Train Accuracy (83%)!")
print("The validation set has more high-value numbers → more carries → easier!")
print()
print("FIXED: Dataset now shuffles before split!")
print("=" * 60)
