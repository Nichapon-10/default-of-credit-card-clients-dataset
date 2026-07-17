# ============================================================
# Streamlit App: Credit Default Prediction — Liquid Glass Edition
# ธีมดำ + การ์ดกระจกฝ้า ตามสไตล์ landing page ตัวอย่าง
# วิธีรัน: streamlit run app.py  (ต้องมี credit_default_model.pkl)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Credit Default AI", page_icon="💳", layout="wide")

# ---------------- Liquid Glass CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Barlow:wght@300;400;500;600&display=swap');

/* พื้นหลังดำ + gradient ดาวจางๆ */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 80% 50% at 50% -10%, rgba(90,110,255,0.18), transparent 60%),
                radial-gradient(ellipse 60% 40% at 85% 110%, rgba(160,80,255,0.12), transparent 60%),
                #000000;
}
[data-testid="stHeader"] { background: transparent; }

html, body, [class*="css"], .stMarkdown, p, label, span {
    font-family: 'Barlow', sans-serif !important;
    color: rgba(255,255,255,0.92);
}

/* หัวข้อใหญ่แบบ Instrument Serif italic */
.hero-title {
    font-family: 'Instrument Serif', serif !important;
    font-style: italic;
    font-size: 4.2rem;
    line-height: 0.95;
    letter-spacing: -2px;
    color: #ffffff;
    text-align: center;
    margin: 0.5rem 0 0.4rem 0;
}
.hero-sub {
    text-align: center;
    color: rgba(255,255,255,0.75);
    font-weight: 300;
    font-size: 0.95rem;
    max-width: 640px;
    margin: 0 auto 0.8rem auto;
}
.hero-badge {
    display: flex; justify-content: center; margin-bottom: 0.8rem;
}
.hero-badge span {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 9999px;
    padding: 6px 16px;
    font-size: 0.78rem;
    color: rgba(255,255,255,0.85);
}
.hero-badge b {
    background: #fff; color: #000;
    border-radius: 9999px; padding: 2px 10px;
    font-size: 0.72rem; margin-right: 8px;
}

/* การ์ดกระจก */
.glass-card {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.14);
    border-top: 1px solid rgba(255,255,255,0.35);
    border-radius: 1.25rem;
    padding: 1.4rem 1.6rem;
    box-shadow: inset 0 1px 1px rgba(255,255,255,0.10);
}
.glass-num {
    font-family: 'Instrument Serif', serif !important;
    font-style: italic; font-size: 2.6rem; line-height: 1; color: #fff;
    letter-spacing: -1px;
}
.glass-label { font-size: 0.78rem; color: rgba(255,255,255,0.7); font-weight: 300; margin-top: 6px; }

/* Section heading */
.section-kicker { color: rgba(255,255,255,0.6); font-size: 0.85rem; margin-bottom: 2px; }
.section-title {
    font-family: 'Instrument Serif', serif !important;
    font-style: italic; font-size: 2.2rem; color: #fff;
    letter-spacing: -1px; line-height: 1; margin-bottom: 0.6rem;
}

/* Tabs เป็นแคปซูลกระจก */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 9999px;
    padding: 5px;
    gap: 4px;
    width: fit-content;
    margin: 0 auto;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9999px !important;
    padding: 6px 18px;
    color: rgba(255,255,255,0.85) !important;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #000000 !important;
}
.stTabs [aria-selected="true"] p { color: #000 !important; font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* Inputs */
[data-testid="stNumberInput"] input, [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 12px !important;
    color: #fff !important;
}
[data-testid="stWidgetLabel"] p { color: rgba(255,255,255,0.8) !important; font-size: 0.82rem; }

/* ปุ่มหลักสีขาว pill */
.stButton > button, .stDownloadButton > button {
    background: #ffffff !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 9999px !important;
    padding: 0.6rem 1.6rem !important;
    font-weight: 600 !important;
    font-family: 'Barlow', sans-serif !important;
    transition: transform .15s ease, box-shadow .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 30px rgba(255,255,255,0.25);
}

/* กล่องผลลัพธ์ */
.result-risk-high  { border-color: rgba(255,90,90,0.5) !important;  box-shadow: inset 0 1px 1px rgba(255,255,255,.1), 0 0 40px rgba(255,60,60,.15); }
.result-risk-mid   { border-color: rgba(255,200,80,0.5) !important; box-shadow: inset 0 1px 1px rgba(255,255,255,.1), 0 0 40px rgba(255,190,60,.12); }
.result-risk-low   { border-color: rgba(120,255,170,0.45) !important; box-shadow: inset 0 1px 1px rgba(255,255,255,.1), 0 0 40px rgba(80,255,160,.12); }

/* Progress bar */
.stProgress > div > div > div { background: linear-gradient(90deg,#8ab4ff,#c58aff) !important; }

/* DataFrame + uploader โทนมืด */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px dashed rgba(255,255,255,0.25) !important;
    border-radius: 1rem !important;
}
hr { border-color: rgba(255,255,255,0.1); }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load('credit_default_model.pkl')


try:
    artifact = load_model()
    model = artifact['model']
    FEATURES = artifact['feature_columns']
except FileNotFoundError:
    st.error("ไม่พบไฟล์ credit_default_model.pkl กรุณาวางไว้โฟลเดอร์เดียวกับ app.py")
    st.stop()

# ---------------- Hero ----------------
st.markdown("""
<div class="hero-badge"><span><b>AI</b>Random Forest · UCI Credit Card Dataset · 30,000 Clients</span></div>
<div class="hero-title">Predict Beyond<br/>the Payment</div>
<p class="hero-sub">ระบบทำนายความเสี่ยงการผิดนัดชำระบัตรเครดิตด้วย Machine Learning
วิเคราะห์จากพฤติกรรมการชำระย้อนหลัง 6 เดือน แม่นยำ 78.8% (AUC 0.77)</p>
""", unsafe_allow_html=True)

# Stats row
s1, s2, s3, s4 = st.columns(4)
for col, num, label in [
    (s1, "30K", "Clients in Training Data"),
    (s2, "78.8%", "Model Accuracy"),
    (s3, "0.77", "ROC – AUC Score"),
    (s4, "23", "Behavioral Features"),
]:
    col.markdown(f'<div class="glass-card" style="text-align:center">'
                 f'<div class="glass-num">{num}</div>'
                 f'<div class="glass-label">{label}</div></div>', unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Single Prediction", "Batch CSV"])

# ---------------- Tab 1 ----------------
with tab1:
    st.markdown('<p class="section-kicker">// Individual Assessment</p>'
                '<p class="section-title">ทำนายรายบุคคล</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("**ข้อมูลส่วนตัว**")
        limit_bal = st.number_input("วงเงินบัตร (LIMIT_BAL)", 10000, 1000000, 100000, step=10000)
        sex = st.selectbox("เพศ", [1, 2], format_func=lambda x: "ชาย" if x == 1 else "หญิง")
        education = st.selectbox("การศึกษา", [1, 2, 3, 4],
                                 format_func=lambda x: {1: "ปริญญาโท/เอก", 2: "ปริญญาตรี",
                                                        3: "มัธยม", 4: "อื่นๆ"}[x])
        marriage = st.selectbox("สถานภาพ", [1, 2, 3],
                                format_func=lambda x: {1: "แต่งงาน", 2: "โสด", 3: "อื่นๆ"}[x])
        age = st.number_input("อายุ", 21, 79, 35)

    pay_options = list(range(-2, 9))
    pay_label = lambda x: {-2: "-2 ไม่มีการใช้จ่าย", -1: "-1 จ่ายเต็ม",
                           0: "0 จ่ายขั้นต่ำ"}.get(x, f"{x} ค้างจ่าย {x} เดือน")

    with col2:
        st.markdown("**สถานะการชำระ 6 เดือน**")
        pay_1 = st.selectbox("เดือนล่าสุด (PAY_1)", pay_options, index=2, format_func=pay_label)
        pay_2 = st.selectbox("2 เดือนก่อน (PAY_2)", pay_options, index=2, format_func=pay_label)
        pay_3 = st.selectbox("3 เดือนก่อน (PAY_3)", pay_options, index=2, format_func=pay_label)
        pay_4 = st.selectbox("4 เดือนก่อน (PAY_4)", pay_options, index=2, format_func=pay_label)
        pay_5 = st.selectbox("5 เดือนก่อน (PAY_5)", pay_options, index=2, format_func=pay_label)
        pay_6 = st.selectbox("6 เดือนก่อน (PAY_6)", pay_options, index=2, format_func=pay_label)

    with col3:
        st.markdown("**ยอดบิลและยอดชำระ**")
        bill_amt1 = st.number_input("ยอดบิลเดือนล่าสุด (BILL_AMT1)", value=50000)
        bill_amt2 = st.number_input("ยอดบิล 2 เดือนก่อน (BILL_AMT2)", value=48000)
        bill_amt3 = st.number_input("ยอดบิล 3 เดือนก่อน (BILL_AMT3)", value=46000)
        bill_avg_rest = st.number_input("ยอดบิลเฉลี่ยเดือน 4-6", value=45000)
        pay_amt1 = st.number_input("ยอดชำระเดือนล่าสุด (PAY_AMT1)", value=2000, min_value=0)
        pay_amt2 = st.number_input("ยอดชำระ 2 เดือนก่อน (PAY_AMT2)", value=2000, min_value=0)
        pay_avg_rest = st.number_input("ยอดชำระเฉลี่ยเดือน 3-6", value=2000, min_value=0)

    st.markdown("<br/>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1, 1])
    predict = mid.button("Start Prediction  ↗", use_container_width=True)

    if predict:
        input_data = pd.DataFrame([{
            'LIMIT_BAL': limit_bal, 'SEX': sex, 'EDUCATION': education,
            'MARRIAGE': marriage, 'AGE': age,
            'PAY_1': pay_1, 'PAY_2': pay_2, 'PAY_3': pay_3,
            'PAY_4': pay_4, 'PAY_5': pay_5, 'PAY_6': pay_6,
            'BILL_AMT1': bill_amt1, 'BILL_AMT2': bill_amt2, 'BILL_AMT3': bill_amt3,
            'BILL_AMT4': bill_avg_rest, 'BILL_AMT5': bill_avg_rest, 'BILL_AMT6': bill_avg_rest,
            'PAY_AMT1': pay_amt1, 'PAY_AMT2': pay_amt2, 'PAY_AMT3': pay_avg_rest,
            'PAY_AMT4': pay_avg_rest, 'PAY_AMT5': pay_avg_rest, 'PAY_AMT6': pay_avg_rest,
        }])[FEATURES]

        proba = float(model.predict_proba(input_data)[0, 1])

        if proba >= 0.5:
            css, verdict, note = "result-risk-high", "High Risk", "มีแนวโน้มผิดนัดชำระเดือนถัดไป — ควรพิจารณามาตรการติดตามหนี้"
        elif proba >= 0.3:
            css, verdict, note = "result-risk-mid", "Moderate Risk", "ความเสี่ยงปานกลาง — ควรติดตามพฤติกรรมการชำระอย่างใกล้ชิด"
        else:
            css, verdict, note = "result-risk-low", "Low Risk", "มีแนวโน้มชำระตามปกติ"

        st.markdown("<br/>", unsafe_allow_html=True)
        r1, r2 = st.columns([1, 2], gap="large")
        with r1:
            st.markdown(f'<div class="glass-card {css}" style="text-align:center">'
                        f'<div class="glass-label">Default Probability</div>'
                        f'<div class="glass-num" style="font-size:3.4rem">{proba:.1%}</div>'
                        f'</div>', unsafe_allow_html=True)
        with r2:
            st.markdown(f'<div class="glass-card {css}">'
                        f'<div class="glass-num" style="font-size:2rem">{verdict}</div>'
                        f'<div class="glass-label" style="font-size:0.9rem">{note}</div>'
                        f'</div>', unsafe_allow_html=True)
            st.progress(proba)

# ---------------- Tab 2 ----------------
with tab2:
    st.markdown('<p class="section-kicker">// Batch Processing</p>'
                '<p class="section-title">ทำนายจากไฟล์ CSV</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:rgba(255,255,255,0.7);font-weight:300">'
                'อัปโหลดไฟล์ CSV ที่มีคอลัมน์เหมือนชุดข้อมูล UCI ระบบจะทำนายทุกแถวและจัดเรียงตามความเสี่ยง</p>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader("เลือกไฟล์ CSV", type="csv", label_visibility="collapsed")

    if uploaded is not None:
        data = pd.read_csv(uploaded)
        data = data.rename(columns={'default.payment.next.month': 'default', 'PAY_0': 'PAY_1'})
        if 'EDUCATION' in data.columns:
            data['EDUCATION'] = data['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
        if 'MARRIAGE' in data.columns:
            data['MARRIAGE'] = data['MARRIAGE'].replace({0: 3})

        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error(f"ไฟล์ขาดคอลัมน์: {missing}")
        else:
            X_batch = data[FEATURES]
            data['โอกาสผิดนัด'] = model.predict_proba(X_batch)[:, 1].round(4)
            data['ผลทำนาย'] = np.where(data['โอกาสผิดนัด'] >= 0.5, 'ผิดนัด', 'จ่ายปกติ')

            n_default = int((data['ผลทำนาย'] == 'ผิดนัด').sum())
            c1, c2, c3 = st.columns(3)
            for col, num, label in [
                (c1, f"{len(data):,}", "Total Records"),
                (c2, f"{n_default:,}", "Predicted Defaults"),
                (c3, f"{n_default/len(data):.1%}", "Default Rate"),
            ]:
                col.markdown(f'<div class="glass-card" style="text-align:center">'
                             f'<div class="glass-num">{num}</div>'
                             f'<div class="glass-label">{label}</div></div>',
                             unsafe_allow_html=True)

            st.markdown("<br/>", unsafe_allow_html=True)
            st.dataframe(data.sort_values('โอกาสผิดนัด', ascending=False),
                         use_container_width=True)

            csv = data.to_csv(index=False).encode('utf-8-sig')
            st.download_button("Download Results  ↓", csv,
                               "prediction_results.csv", "text/csv")

st.markdown('<br/><p style="text-align:center;color:rgba(255,255,255,0.4);font-size:0.75rem;font-weight:300">'
            'ผลทำนายเป็นเครื่องมือช่วยประเมินความเสี่ยง ไม่ใช่คำตัดสินทางการเงิน · '
            'Aeon · Vela · Apex · Orbit · Zeno</p>', unsafe_allow_html=True)
