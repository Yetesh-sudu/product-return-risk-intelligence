import pandas as pd
import numpy as np

# Reproducible random data
np.random.seed(42)

# Number of orders
n = 5000

# -----------------------------
# BASIC ORDER DATA
# -----------------------------

order_ids = [f"ORD{i:05d}" for i in range(1, n + 1)]

customer_ids = [
    f"CUST{np.random.randint(1, 1501):04d}"
    for _ in range(n)
]

categories = np.random.choice(
    ["Clothing", "Electronics", "Footwear",
     "Beauty", "Home", "Sports"],
    n
)

product_price = np.round(
    np.random.uniform(200, 50000, n), 2
)

discount = np.random.choice(
    [0, 5, 10, 15, 20, 25, 30, 40, 50],
    n
)

quantity = np.random.randint(1, 5, n)

customer_age = np.random.randint(18, 66, n)

gender = np.random.choice(
    ["Male", "Female", "Other"],
    n,
    p=[0.48, 0.48, 0.04]
)

payment_method = np.random.choice(
    ["UPI", "COD", "Credit Card", "Debit Card", "Net Banking"],
    n
)

shipping_days = np.random.randint(1, 11, n)

previous_orders = np.random.randint(1, 31, n)

previous_returns = np.array([
    np.random.randint(0, order + 1)
    for order in previous_orders
])

customer_rating = np.random.randint(1, 6, n)

product_rating = np.random.randint(1, 6, n)


# -----------------------------
# CREATE RETURN RISK
# -----------------------------

risk = np.full(n, 0.10)

# Previous return behaviour
return_ratio = previous_returns / previous_orders
risk += return_ratio * 0.35

# Large discounts
risk += (discount >= 30) * 0.10

# Slow shipping
risk += (shipping_days >= 7) * 0.10

# Poor product rating
risk += (product_rating <= 2) * 0.12

# Clothing and footwear generally have
# higher size/fit-related return risk
risk += np.isin(
    categories,
    ["Clothing", "Footwear"]
) * 0.08

# COD orders
risk += (payment_method == "COD") * 0.05

# Keep probability within sensible range
risk = np.clip(risk, 0.05, 0.90)

return_status = np.random.binomial(
    1,
    risk
)


# -----------------------------
# CREATE DATAFRAME
# -----------------------------

df = pd.DataFrame({
    "Order_ID": order_ids,
    "Customer_ID": customer_ids,
    "Product_Category": categories,
    "Product_Price": product_price,
    "Discount_Percentage": discount,
    "Quantity": quantity,
    "Customer_Age": customer_age,
    "Customer_Gender": gender,
    "Payment_Method": payment_method,
    "Shipping_Days": shipping_days,
    "Previous_Orders": previous_orders,
    "Previous_Returns": previous_returns,
    "Customer_Rating": customer_rating,
    "Product_Rating": product_rating,
    "Return_Status": return_status
})


# -----------------------------
# SAVE DATASET
# -----------------------------

df.to_csv(
    "data/product_returns.csv",
    index=False
)

print("Dataset created successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nReturn distribution:")
print(df["Return_Status"].value_counts())

print("\nReturn percentage:")
print(
    df["Return_Status"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)