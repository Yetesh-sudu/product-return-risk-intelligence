def get_risk_level(probability):
    """
    Convert return probability into a risk category.
    """

    if probability < 0.35:
        return "Low"

    elif probability < 0.65:
        return "Medium"

    else:
        return "High"


def get_risk_factors(data):
    """
    Identify understandable factors that may contribute
    to product return risk.
    """

    factors = []

    # Previous return behaviour
    if data["Previous_Return_Ratio"] >= 0.50:
        factors.append(
            "Customer has a high previous return ratio."
        )

    elif data["Previous_Return_Ratio"] >= 0.30:
        factors.append(
            "Customer has a moderate previous return history."
        )

    # Discount
    if data["Discount_Percentage"] >= 30:
        factors.append(
            "Product has a high discount."
        )

    # Shipping
    if data["Shipping_Days"] >= 7:
        factors.append(
            "Long shipping time may increase return risk."
        )

    # Product rating
    if data["Product_Rating"] <= 2:
        factors.append(
            "Product has a low customer rating."
        )

    # Product category
    if data["Product_Category"] in [
        "Clothing",
        "Footwear"
    ]:
        factors.append(
            "Product belongs to a category with higher return tendency."
        )

    # Payment method
    if data["Payment_Method"] == "COD":
        factors.append(
            "Cash on Delivery orders show slightly higher return tendency."
        )

    if not factors:
        factors.append(
            "No major return-risk indicators were detected."
        )

    return factors
if __name__ == "__main__":

    sample = {
        "Previous_Return_Ratio": 0.60,
        "Discount_Percentage": 40,
        "Shipping_Days": 9,
        "Product_Rating": 2,
        "Product_Category": "Clothing",
        "Payment_Method": "COD"
    }

    probability = 0.76

    print(
        "Risk Level:",
        get_risk_level(probability)
    )

    print("\nRisk Factors:")

    for factor in get_risk_factors(sample):
        print("-", factor)