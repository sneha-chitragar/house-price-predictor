predicted_price = int(prediction[0])

st.success(
    f"💰 Estimated Property Price: ₹ {predicted_price:,}"
)


# -----------------------------
# NEW GRAPH (INPUT + PREDICTION)
# -----------------------------

st.subheader("📊 Property Details Visualization")


numeric_features = []
numeric_values = []


for key, value in input_data.items():

    try:
        val = float(value)

        numeric_features.append(
            key.replace("_", " ").title()
        )

        numeric_values.append(val)

    except:
        pass  # ignore text values like location


# Add predicted price
numeric_features.append("Predicted Price")
numeric_values.append(predicted_price)


chart_df = pd.DataFrame({
    "Feature": numeric_features,
    "Value": numeric_values
})


# (Optional Debug - remove later)
# st.write(chart_df)


fig, ax = plt.subplots(figsize=(6,3))


ax.bar(
    chart_df["Feature"],
    chart_df["Value"]
)


ax.set_ylabel("Value")
ax.set_title("Input Features + Prediction")


plt.xticks(rotation=45, ha="right")


plt.tight_layout()


st.pyplot(fig, use_container_width=False)