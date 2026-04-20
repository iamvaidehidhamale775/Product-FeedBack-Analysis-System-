import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("📊 Product Feedback Analysis System")

# ---------- MODEL ----------
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

model = load_model()

# ---------- UPLOAD ----------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'review' not in df.columns:
        st.error("CSV must contain 'review' column")
        st.stop()

    # ---------- SENTIMENT ----------
    results = model(df['review'].tolist())
    df['sentiment'] = [r['label'] for r in results]
    df['confidence'] = [r['score'] for r in results]

    df['sentiment'] = df['sentiment'].map({
        "POSITIVE": "Positive",
        "NEGATIVE": "Negative"
    })

    # ---------- CATEGORY ----------
    def categorize(text):
        text = text.lower()
        if "price" in text:
            return "Price"
        elif "delivery" in text:
            return "Delivery"
        elif "package" in text:
            return "Packaging"
        else:
            return "Quality"

    df['category'] = df['review'].apply(categorize)

    # ---------- DATE ----------
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # ---------- PRODUCT FILTER ----------
    if 'product' in df.columns:
        selected_product = st.selectbox("Filter by Product", ["All"] + list(df['product'].unique()))

        if selected_product != "All":
            df = df[df['product'] == selected_product]

    # ---------- KPIs ----------
    total = len(df)
    pos = (df['sentiment'] == "Positive").sum()
    neg = (df['sentiment'] == "Negative").sum()

    score = round((pos / total) * 5, 2)

    col1, col2, col3 = st.columns([2, 2, 1])

    # ---------- GAUGE ----------
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': "Score (out of 5)"},
            gauge={'axis': {'range': [0, 5]},
                   'bar': {'color': "green"}}
        ))
        st.plotly_chart(fig, use_container_width=True)

    # ---------- DONUT ----------
    with col2:
        fig2 = px.pie(
            df,
            names='sentiment',
            hole=0.6,
            color='sentiment',
            color_discrete_map={
                "Positive": "#a855f7",
                "Negative": "#ec4899"
            }
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ---------- METRICS ----------
    with col3:
        st.metric("Total Reviews", total)
        st.metric("Positive", pos)
        st.metric("Negative", neg)

    # ---------- CATEGORY ----------
    st.subheader("📊 Category Distribution")

    cat_counts = df['category'].value_counts().reset_index()
    cat_counts.columns = ['Category', 'Count']

    fig_cat = px.bar(cat_counts, x='Category', y='Count', color='Category')
    st.plotly_chart(fig_cat, use_container_width=True)

    # ---------- CATEGORY vs SENTIMENT ----------
    st.subheader("📊 Category vs Sentiment")

    cat_sent = df.groupby(['category', 'sentiment']).size().reset_index(name='count')

    fig_cs = px.bar(cat_sent, x='category', y='count', color='sentiment')
    st.plotly_chart(fig_cs, use_container_width=True)

    # ---------- 📈 MONTHLY TIMELINE (FIXED) ----------
    if 'date' in df.columns:
        st.subheader("📈 Monthly Sentiment Analysis")

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['month'] = df['date'].dt.to_period('M').astype(str)

        monthly = df.groupby(['month', 'sentiment']).size().reset_index(name='count')

        fig3 = px.line(
            monthly,
            x='month',
            y='count',
            color='sentiment',
            markers=True,
            color_discrete_map={
                "Positive": "#a855f7",
                "Negative": "#ec4899"
            }
        )

        st.plotly_chart(fig3, use_container_width=True)

    # ---------- PRODUCT RANKING ----------
    if 'product' in df.columns:
        st.subheader("🏆 Product Ranking")

        product_stats = df.groupby('product').apply(
            lambda x: (x['sentiment'] == "Positive").sum() / len(x)
        ).reset_index(name='positive_ratio')

        product_stats = product_stats.sort_values(by='positive_ratio', ascending=False)

        best = product_stats.iloc[0]
        worst = product_stats.iloc[-1]

        c1, c2 = st.columns(2)

        with c1:
            st.success(f"🥇 Best Product: {best['product']} ({best['positive_ratio']*100:.1f}%)")

        with c2:
            st.error(f"⚠️ Worst Product: {worst['product']} ({worst['positive_ratio']*100:.1f}%)")

        st.dataframe(product_stats)

    

    # ---------- INSIGHTS ----------
    st.subheader("💡 Insights")

    insights = []

    if pos > neg:
        insights.append("Customers are generally satisfied with the product.")
    else:
        insights.append("Customers are generally dissatisfied with the product.")

    if df[df['category'] == "Delivery"].shape[0] > 0:
        insights.append("Delivery-related issues are common.")

    if df[df['category'] == "Price"].shape[0] > 0:
        insights.append("Customers have concerns about pricing.")

    for i in insights:
        st.write("•", i)

    

    # ---------- DATA ----------
    st.subheader("📄 Data")
    st.dataframe(df)

    # ---------- DOWNLOAD ----------
    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        file_name="analyzed_data.csv"
    )