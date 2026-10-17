import pandas as pd

# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv("data/product_returns.csv")

print("Original dataset shape:", df.shape)


# -----------------------------
# CHECK MISSING VALUES
# -----------------------------

print("\nMissing values:")
print(df.isnull().sum())


# -----------------------------
# REMOVE DUPLICATES
# -----------------------------

df = df.drop_duplicates()

print("\nShape after removing duplicates:", df.shape)


# -----------------------------
# FEATURE ENGINEERING
# -----------------------------

# Customer's historical return rate
df["Previous_Return_Ratio"] = (
    df["Previous_Returns"] /
    df["Previous_Orders"]
)

# Order value before discount
df["Order_Value"] = (
    df["Product_Price"] *
    df["Quantity"]
)

# Effective value after discount
df["Discounted_Order_Value"] = (
    df["Order_Value"] *
    (1 - df["Discount_Percentage"] / 100)
)


# -----------------------------
# DROP IDENTIFIER COLUMNS
# -----------------------------

df = df.drop(
    columns=[
        "Order_ID",
        "Customer_ID"
    ]
)


# -----------------------------
# SAVE PROCESSED DATA
# -----------------------------

df.to_csv(
    "data/processed_returns.csv",
    index=False
)

print("\nPreprocessing completed!")
print("Processed dataset shape:", df.shape)

print("\nProcessed columns:")
print(df.columns.tolist())