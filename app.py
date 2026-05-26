import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Social Media Sentiment Analyzer",
    page_icon="🧠",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .stTextArea textarea {
        background-color: #1e293b;
        color: #e2e8f0;
        border-radius: 10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .positive { border-left: 4px solid #22c55e; }
    .negative { border-left: 4px solid #ef4444; }
    .neutral  { border-left: 4px solid #f59e0b; }
    h1 { color: #f8fafc; }
    h2, h3 { color: #cbd5e1; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

# ── Label Map ─────────────────────────────────────────────────────────────────
LABEL_MAP = {
    "positive": ("😊 Positive", "#22c55e", "positive"),
    "negative": ("😠 Negative", "#ef4444", "negative"),
    "neutral":  ("😐 Neutral",  "#f59e0b", "neutral"),
}

def get_label(raw_label):
    raw = raw_label.lower()
    for key in LABEL_MAP:
        if key in raw:
            return LABEL_MAP[key]
    return (raw_label, "#94a3b8", "neutral")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧠 Social Media Sentiment Analyzer")
st.markdown("**Powered by RoBERTa — trained on 58M tweets**")
st.divider()

# ── Load Model with Spinner ───────────────────────────────────────────────────
with st.spinner("Loading AI model (first run may take ~30 seconds)..."):
    classifier = load_model()
st.success("✅ Model ready!")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📝 Single Post Analysis", "📊 Bulk Analysis"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Single Post
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Enter a Social Media Post")
        example_posts = [
            "I absolutely love how this new feature works! 🚀",
            "This is the worst product I've ever used. Total disaster.",
            "Just got the package today. It arrived on time.",
            "OMG can't believe how amazing this concert was!! 🎶❤️",
            "Terrible customer service, waited 2 hours for nothing.",
        ]
        selected = st.selectbox(
            "Or try an example:",
            ["-- type your own --"] + example_posts
        )
        user_input = st.text_area(
            "Post text:",
            value="" if selected == "-- type your own --" else selected,
            height=150,
            placeholder="Type a tweet, comment, or post here..."
        )
        analyze_btn = st.button(
            "🔍 Analyze Sentiment",
            type="primary",
            use_container_width=True
        )

    with col2:
        st.subheader("Result")
        if analyze_btn and user_input.strip():
            with st.spinner("Analyzing..."):
                result = classifier(user_input[:512])[0]
            label_text, color, css_class = get_label(result["label"])
            confidence = result["score"] * 100

            st.markdown(f"""
            <div class="metric-card {css_class}">
                <h1 style="font-size:3rem;margin:0">{label_text}</h1>
                <p style="color:#94a3b8;margin-top:8px">Confidence</p>
                <h2 style="color:{color};margin:0">{confidence:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Confidence Score", "font": {"color": "#e2e8f0"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                    "bar": {"color": color},
                    "bgcolor": "#1e293b",
                    "steps": [
                        {"range": [0, 50],  "color": "#1e293b"},
                        {"range": [50, 100], "color": "#0f172a"},
                    ],
                },
                number={"suffix": "%", "font": {"color": "#e2e8f0"}}
            ))
            fig.update_layout(
                paper_bgcolor="#0f172a",
                font_color="#e2e8f0",
                height=250
            )
            st.plotly_chart(fig, use_container_width=True)

        elif analyze_btn:
            st.warning("Please enter some text first.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Bulk Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Analyze Multiple Posts at Once")
    bulk_text = st.text_area(
        "Paste posts (one per line):",
        height=200,
        placeholder="Great product!\nTerrible experience.\nJust okay I guess.\nAbsolutely love it!",
    )
    bulk_btn = st.button(
        "🚀 Analyze All Posts",
        type="primary",
        use_container_width=True
    )

    if bulk_btn and bulk_text.strip():
        posts = [p.strip() for p in bulk_text.strip().split("\n") if p.strip()]

        with st.spinner(f"Analyzing {len(posts)} posts..."):
            results = classifier(posts, truncation=True, max_length=512)

        rows = []
        for post, res in zip(posts, results):
            label_text, color, _ = get_label(res["label"])
            rows.append({
                "Post": post,
                "Sentiment": label_text,
                "Confidence": f"{res['score']*100:.1f}%",
                "_score": res["score"],
                "_raw_label": res["label"].lower()
            })

        df = pd.DataFrame(rows)

        # Summary Metrics
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        total = len(df)
        pos = sum("positive" in r for r in df["_raw_label"])
        neg = sum("negative" in r for r in df["_raw_label"])
        neu = total - pos - neg

        m1.metric("Total Posts", total)
        m2.metric("😊 Positive", f"{pos} ({pos/total*100:.0f}%)")
        m3.metric("😠 Negative", f"{neg} ({neg/total*100:.0f}%)")
        m4.metric("😐 Neutral",  f"{neu} ({neu/total*100:.0f}%)")

        # Charts
        col_a, col_b = st.columns(2)
        pie_df = pd.DataFrame({
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Count": [pos, neg, neu]
        })

        with col_a:
            fig_pie = px.pie(
                pie_df, names="Sentiment", values="Count",
                color="Sentiment",
                color_discrete_map={
                    "Positive": "#22c55e",
                    "Negative": "#ef4444",
                    "Neutral":  "#f59e0b"
                },
                title="Sentiment Distribution"
            )
            fig_pie.update_layout(
                paper_bgcolor="#0f172a",
                font_color="#e2e8f0"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            fig_bar = px.bar(
                pie_df, x="Sentiment", y="Count",
                color="Sentiment",
                color_discrete_map={
                    "Positive": "#22c55e",
                    "Negative": "#ef4444",
                    "Neutral":  "#f59e0b"
                },
                title="Post Count by Sentiment"
            )
            fig_bar.update_layout(
                paper_bgcolor="#0f172a",
                font_color="#e2e8f0",
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Results Table
        st.subheader("📋 Detailed Results")
        st.dataframe(
            df[["Post", "Sentiment", "Confidence"]],
            use_container_width=True,
            hide_index=True
        )

        # Download Button
        csv = df[["Post", "Sentiment", "Confidence"]].to_csv(index=False)
        st.download_button(
            "⬇️ Download Results as CSV",
            csv,
            "sentiment_results.csv",
            "text/csv"
        )

    elif bulk_btn:
        st.warning("Please enter at least one post.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:#475569'>Model: cardiffnlp/twitter-roberta-base-sentiment-latest | Built with Streamlit & HuggingFace</p>",
    unsafe_allow_html=True
          )
