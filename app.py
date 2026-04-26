import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px
import plotly.graph_objects as go

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from io import BytesIO
import plotly.io as pio
from datetime import datetime

st.set_page_config(layout="wide")
st.title("📊 Product Feedback Analysis")

# ---------- MODEL ----------
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

model = load_model()

# ---------- FIG TO IMAGE ----------
def fig_to_image(fig):
    try:
        img_bytes = pio.to_image(fig, format="png")
        return BytesIO(img_bytes)
    except:
        return None

# ---------- PDF ----------
def generate_report(total, pos, neg, score, insights,
                    fig2, fig_cat, fig_cs, fig3,
                    best=None, worst=None):

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    # 🎨 Styles
    title_style = ParagraphStyle(
        'title', parent=styles['Title'],
        textColor=colors.HexColor("#6D28D9"),
        spaceAfter=20
    )

    section_style = ParagraphStyle(
        'section', parent=styles['Heading2'],
        textColor=colors.HexColor("#2563EB"),
        spaceAfter=10
    )

    normal = styles['Normal']

    content = []

    # ---------- COVER PAGE ----------
    content.append(Spacer(1, 200))
    content.append(Paragraph("Product Feedback Analysis System", title_style))
    content.append(Spacer(1, 20))
    content.append(Paragraph("Project Report", styles['Heading2']))
    content.append(Spacer(1, 40))
    content.append(Paragraph(f"Generated on: {datetime.now().strftime('%d %B %Y')}", normal))
    content.append(PageBreak())

    # ---------- INDEX ----------
    content.append(Paragraph("Table of Contents", title_style))
    content.append(Paragraph("1. Overview", normal))
    content.append(Paragraph("2. Sentiment Distribution", normal))
    content.append(Paragraph("3. Category Distribution", normal))
    content.append(Paragraph("4. Category vs Sentiment", normal))
    content.append(Paragraph("5. Monthly Trend", normal))
    content.append(Paragraph("6. Product Insights", normal))
    content.append(Paragraph("7. Key Insights", normal))
    content.append(PageBreak())

    # ---------- OVERVIEW ----------
    content.append(Paragraph("1. Overview", section_style))
    content.append(Paragraph(f"Total Reviews: {total}", normal))
    content.append(Paragraph(f"Positive Reviews: {pos}", normal))
    content.append(Paragraph(f"Negative Reviews: {neg}", normal))
    content.append(Paragraph(f"Score (out of 5): {score}", normal))
    content.append(Spacer(1, 20))

    # ---------- CHARTS ----------
    sections = [
        ("2. Sentiment Distribution", fig2),
        ("3. Category Distribution", fig_cat),
        ("4. Category vs Sentiment", fig_cs),
        ("5. Monthly Trend", fig3)
    ]

    for title, fig in sections:
        if fig:
            img = fig_to_image(fig)
            if img:
                content.append(Paragraph(title, section_style))
                content.append(Image(img, width=5*inch, height=3*inch))
                content.append(Spacer(1, 20))

    # ---------- PRODUCT INSIGHTS ----------
    if best is not None and worst is not None:
        content.append(Paragraph("6. Product Insights", section_style))
        content.append(Paragraph(f"Best Product: {best['product']}", normal))
        content.append(Paragraph(f"Worst Product: {worst['product']}", normal))
        content.append(Spacer(1, 20))

    # ---------- INSIGHTS ----------
    content.append(Paragraph("7. Key Insights", section_style))
    for i in insights:
        content.append(Paragraph(f"• {i}", normal))

    content.append(Spacer(1, 20))

    footer = ParagraphStyle('footer', parent=styles['Normal'],
                            textColor=colors.grey, fontSize=8)

    content.append(Paragraph("Generated using Product Feedback Analysis System", footer))

    doc.build(content)
    buffer.seek(0)
    return buffer

# ---------- FILE ----------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'review' not in df.columns:
        st.error("CSV must contain 'review' column")
        st.stop()

    # SENTIMENT
    results = model(df['review'].tolist())
    df['sentiment'] = [r['label'] for r in results]
    df['confidence'] = [r['score'] for r in results]

    df['sentiment'] = df['sentiment'].map({
        "POSITIVE": "Positive",
        "NEGATIVE": "Negative"
    })

    # CATEGORY
    def categorize(text):
        text = str(text).lower()
        return "Price" if "price" in text else "Quality"

    df['category'] = df['review'].apply(categorize)

    # DATE
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # FILTER
    if 'product' in df.columns:
        selected = st.selectbox("Filter by Product", ["All"] + list(df['product'].dropna().unique()))
        if selected != "All":
            df = df[df['product'] == selected]

    # KPIs
    total = len(df)
    pos = (df['sentiment'] == "Positive").sum()
    neg = (df['sentiment'] == "Negative").sum()
    score = round((pos / total) * 5, 2) if total else 0

    col1, col2, col3 = st.columns([2,2,1])

    # GAUGE
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': "Score (out of 5)"},
            gauge={'axis': {'range': [0,5]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

    # DONUT
    with col2:
        fig2 = px.pie(df, names='sentiment', hole=0.6,
                      color='sentiment',
                      color_discrete_map={"Positive":"#22c55e","Negative":"#ef4444"})
        st.plotly_chart(fig2, use_container_width=True)

    # METRICS
    with col3:
        st.metric("Total", total)
        st.metric("Positive", pos)
        st.metric("Negative", neg)

    # CATEGORY
    st.subheader("📊 Category Distribution")
    cat = df['category'].value_counts().reset_index()
    cat.columns = ['Category','Count']
    fig_cat = px.bar(cat, x='Category', y='Count', color='Category')
    st.plotly_chart(fig_cat, use_container_width=True)

    # CATEGORY VS SENTIMENT
    st.subheader("📊 Category vs Sentiment")
    cs = df.groupby(['category','sentiment']).size().reset_index(name='count')
    fig_cs = px.bar(cs, x='category', y='count', color='sentiment')
    st.plotly_chart(fig_cs, use_container_width=True)

    # MONTHLY
    fig3 = None
    if 'date' in df.columns:
        st.subheader("📈 Monthly Sentiment Analysis")
        df['month'] = df['date'].dt.to_period('M').astype(str)
        monthly = df.groupby(['month','sentiment']).size().reset_index(name='count')
        fig3 = px.line(monthly, x='month', y='count', color='sentiment', markers=True)
        st.plotly_chart(fig3, use_container_width=True)


    # RANKING
    best = worst = None
    if 'product' in df.columns:
        st.subheader("🏆 Product Ranking")
        rank = df.groupby('product').apply(
            lambda x: (x['sentiment']=="Positive").sum()/len(x)
        ).reset_index(name='score').sort_values(by='score', ascending=False)

        best = rank.iloc[0]
        worst = rank.iloc[-1]

        c1,c2 = st.columns(2)
        c1.success(f"Best: {best['product']}")
        c2.error(f"Worst: {worst['product']}")

        st.dataframe(rank)

    # INSIGHTS
    st.subheader("💡 Insights")
    insights = []
    insights.append(f"Total reviews analyzed: {total}")
    insights.append("Overall sentiment is positive" if pos > neg else "Overall sentiment is negative")

    if neg > 0:
        top_issue = df[df['sentiment']=="Negative"]['category'].value_counts().idxmax()
        insights.append(f"Major issue area: {top_issue}")

    if best is not None and worst is not None:
        insights.append(f"Best product: {best['product']}")
        insights.append(f"Worst product: {worst['product']}")

    for i in insights:
        st.write("•", i)

    # SUMMARY
    st.subheader("🧾 Summary")
    st.success(" ".join(insights))

    # DOWNLOAD CSV
    st.download_button("Download CSV", df.to_csv(index=False), "data.csv")

    # DOWNLOAD PDF
    pdf = generate_report(
        total, pos, neg, score, insights,
        fig2, fig_cat, fig_cs, fig3,
        best, worst
    )

    st.download_button(
        "📄 Download Final Report",
        data=pdf,
        file_name="final_report.pdf",
        mime="application/pdf"
    )
