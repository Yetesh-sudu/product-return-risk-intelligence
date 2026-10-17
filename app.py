import streamlit as st
import pandas as pd
import joblib

from risk_intelligence import get_risk_level, get_risk_factors


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Product Return Risk Intelligence",
    page_icon="📦",
    layout="wide"
)


# --------------------------------
# LOAD MODEL
# --------------------------------

@st.cache_resource
def load_model():
    return joblib.load(
        "models/best_return_risk_model.pkl"
    )


model = load_model()


# --------------------------------
# HEADER
# --------------------------------

st.title("📦 Product Return Behaviour & Return-Risk Intelligence")

st.write(
    "Machine Learning system for predicting the probability "
    "of an e-commerce product being returned."
)

st.divider()


# --------------------------------
# SIDEBAR
# --------------------------------

st.sidebar.header("About the Project")

st.sidebar.write(
    """
    This application analyzes customer, product and order
    information to estimate product return risk.

    The system provides:

    • Return probability  
    • Risk classification  
    • Risk indicators  
    • Data-driven return intelligence
    """
)


# --------------------------------
# INPUT SECTION
# --------------------------------

st.header("🛒 Order Information")

col1, col2, col3 = st.columns(3)


with col1:

    product_category = st.selectbox(
        "Product Category",
        [
            "Clothing",
            "Electronics",
            "Footwear",
            "Beauty",
            "Home",
            "Sports"
        ]
    )

    product_price = st.number_input(
        "Product Price (₹)",
        min_value=100.0,
        max_value=100000.0,
        value=2500.0
    )

    quantity = st.number_input(
        "Quantity",
        min_value=1,
        max_value=10,
        value=1
    )

    discount = st.slider(
        "Discount Percentage",
        0,
        50,
        10
    )


with col2:

    customer_age = st.slider(
        "Customer Age",
        18,
        80,
        30
    )

    customer_gender = st.selectbox(
        "Customer Gender",
        [
            "Male",
            "Female",
            "Other"
        ]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "UPI",
            "COD",
            "Credit Card",
            "Debit Card",
            "Net Banking"
        ]
    )

    shipping_days = st.slider(
        "Shipping Days",
        1,
        15,
        4
    )


with col3:

    previous_orders = st.number_input(
        "Previous Orders",
        min_value=1,
        max_value=100,
        value=10
    )

    previous_returns = st.number_input(
        "Previous Returns",
        min_value=0,
        max_value=int(previous_orders),
        value=2
    )

    customer_rating = st.slider(
        "Customer Rating",
        1,
        5,
        4
    )

    product_rating = st.slider(
        "Product Rating",
        1,
        5,
        4
    )


# --------------------------------
# FEATURE ENGINEERING
# --------------------------------

previous_return_ratio = (
    previous_returns /
    previous_orders
)

order_value = (
    product_price *
    quantity
)

discounted_order_value = (
    order_value *
    (1 - discount / 100)
)


# --------------------------------
# CREATE INPUT DATAFRAME
# --------------------------------

input_data = pd.DataFrame({
    "Product_Category": [product_category],
    "Product_Price": [product_price],
    "Discount_Percentage": [discount],
    "Quantity": [quantity],
    "Customer_Age": [customer_age],
    "Customer_Gender": [customer_gender],
    "Payment_Method": [payment_method],
    "Shipping_Days": [shipping_days],
    "Previous_Orders": [previous_orders],
    "Previous_Returns": [previous_returns],
    "Customer_Rating": [customer_rating],
    "Product_Rating": [product_rating],
    "Previous_Return_Ratio": [previous_return_ratio],
    "Order_Value": [order_value],
    "Discounted_Order_Value": [discounted_order_value]
})


# --------------------------------
# PREDICTION
# --------------------------------

st.divider()

if st.button(
    "🔍 Predict Return Risk",
    use_container_width=True
):

    probability = model.predict_proba(
        input_data
    )[0][1]

    risk_level = get_risk_level(
        probability
    )

    risk_data = {
        "Previous_Return_Ratio":
            previous_return_ratio,

        "Discount_Percentage":
            discount,

        "Shipping_Days":
            shipping_days,

        "Product_Rating":
            product_rating,

        "Product_Category":
            product_category,

        "Payment_Method":
            payment_method
    }

    factors = get_risk_factors(
        risk_data
    )


    # --------------------------------
    # RESULTS
    # --------------------------------

    st.header("📊 Return-Risk Analysis")

    result1, result2, result3 = st.columns(3)

    with result1:

        st.metric(
            "Return Probability",
            f"{probability * 100:.1f}%"
        )

    with result2:

        st.metric(
            "Risk Level",
            risk_level
        )

    with result3:

        st.metric(
            "Previous Return Rate",
            f"{previous_return_ratio * 100:.1f}%"
        )


    # --------------------------------
    # RISK MESSAGE
    # --------------------------------

    if risk_level == "High":

        st.error(
            "🔴 HIGH RETURN RISK"
        )

    elif risk_level == "Medium":

        st.warning(
            "🟠 MEDIUM RETURN RISK"
        )

    else:

        st.success(
            "🟢 LOW RETURN RISK"
        )


    # --------------------------------
    # PROBABILITY BAR
    # --------------------------------

    st.subheader("Return Risk Score")

    st.progress(
        int(probability * 100)
    )


    # --------------------------------
    # RISK FACTORS
    # --------------------------------

    st.subheader("⚠️ Risk Indicators")

    for factor in factors:

        st.write(
            "•", factor
        )


    # --------------------------------
    # BUSINESS RECOMMENDATION
    # --------------------------------

    st.subheader("💡 Business Recommendation")

    if risk_level == "High":

        st.write(
            """
            This order requires additional attention.
            Consider verifying product expectations,
            delivery estimates and customer return history
            before fulfilment.
            """
        )

    elif risk_level == "Medium":

        st.write(
            """
            This order has moderate return risk.
            Monitor the identified risk indicators and
            provide clear product and delivery information.
            """
        )

    else:

        st.write(
            """
            This order currently shows relatively low
            return risk based on the available information.
            """
        )


st.divider()

st.caption(
    "Product Return Behaviour & Return-Risk Intelligence | "
    "Machine Learning Project"
)