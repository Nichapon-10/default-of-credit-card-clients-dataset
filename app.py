# ============================================================
# Streamlit App: ระบบประเมินความเสี่ยงผิดนัดชำระบัตรเครดิต
# ธีมสว่าง อ่านสบายตา ภาษาไทยทั้งหมด
# วิธีรัน: streamlit run app.py  (ต้องมี credit_default_model.pkl)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="ประเมินความเสี่ยงบัตรเครดิต",
                   page_icon="💳", layout="wide")

# ---------------- ธีมสว่าง สไตล์ไทย ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@400;500;600;700&family=Sarabun:wght@300;400;500;600&display=swap');

/* พื้นหลังฟ้าอ่อนสบายตา */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #EEF4FF 0%, #F8FAFD 40%, #FFFFFF 100%);
}
[data-testid="stHeader"] { background: transparent; }

html, body, .stMarkdown, p, span, label, div {
    font-family: 'Sarabun', sans-serif;
    color: #1E2A3B;
}
h1, h2, h3, .heading { font-family: 'Prompt', sans-serif !important; }

/* หัวเว็บ */
.hero-wrap { text-align: center; padding: 0.5rem 0 0.2rem 0; }
.hero-title {
    font-family: 'Prompt', sans-serif !important;
    font-size: 2.6rem; font-weight: 700; color: #14337C;
    line-height: 1.25; margin: 0;
}
.hero-sub {
    font-size: 1.02rem; color: #46608A; font-weight: 400;
    max-width: 720px; margin: 0.5rem auto 0 auto;
}
.hero-chip {
    display: inline-block; background: #FFFFFF; color: #14337C;
    border: 1.5px solid #C9DAF5; border-radius: 999px;
    padding: 6px 18px; font-size: 0.88rem; font-weight: 500;
    box-shadow: 0 2px 8px rgba(20,51,124,0.06); margin-bottom: 0.7rem;
}

/* การ์ดสถิติ */
.stat-card {
    background: #FFFFFF; border: 1px solid #E3EBF7;
    border-radius: 16px; padding: 1.1rem 1rem; text-align: center;
    box-shadow: 0 4px 14px rgba(20,51,124,0.06);
}
.stat-num {
    font-family: 'Prompt', sans-serif !important;
    font-size: 2rem; font-weight: 700; color: #1D4ED8; line-height: 1.1;
}
.stat-label { font-size: 0.86rem; color: #5B7194; margin-top: 4px; }

/* หัวข้อย่อยแต่ละส่วน */
.sec-title {
    font-family: 'Prompt', sans-serif !important;
    font-size: 1.35rem; font-weight: 600; color: #14337C;
    border-left: 5px solid #2563EB; padding-left: 12px;
    margin: 0.4rem 0 0.6rem 0;
}
.group-head {
    font-family: 'Prompt', sans-serif !important;
    font-weight: 600; font-size: 1.02rem; color: #1E2A3B;
    background: #EAF1FE; border-radius: 10px;
    padding: 8px 14px; margin-bottom: 10px;
}

/* แท็บ */
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF; border: 1px solid #DDE7F5;
    border-radius: 999px; padding: 5px; gap: 4px;
    width: fit-content; margin: 0 auto;
    box-shadow: 0 2px 10px rgba(20,51,124,0.05);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 999px !important; padding: 8px 22px;
    color: #46608A !important; font-family: 'Prompt', sans-serif !important;
}
.stTabs [aria-selected="true"] { background: #2563EB !important; }
.stTabs [aria-selected="true"] p { color: #FFFFFF !important; font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* ช่องกรอก */
[data-testid="stNumberInput"] input, [data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border: 1.5px solid #CBD9EE !important;
    border-radius: 10px !important;
    color: #1E2A3B !important;
}
[data-testid="stWidgetLabel"] p {
    color: #33475F !important; font-size: 0.9rem; font-weight: 500;
}

/* ปุ่มหลักชัดเจน */
.stButton > button {
    background: #2563EB !important; color: #FFFFFF !important;
    border: none !important; border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    font-family: 'Prompt', sans-serif !important;
    font-size: 1.05rem !important; font-weight: 600 !important;
    box-shadow: 0 6px 16px rgba(37,99,235,0.30);
    transition: transform .12s ease, box-shadow .12s ease;
}
.stButton > button:hover {
    background: #1D4ED8 !important;
    transform: translateY(-1px);
    box-shadow: 0 8px 22px rgba(37,99,235,0.38);
}
.stDownloadButton > button {
    background: #FFFFFF !important; color: #1D4ED8 !important;
    border: 2px solid #2563EB !important; border-radius: 12px !important;
    font-family: 'Prompt', sans-serif !important; font-weight: 600 !important;
    padding: 0.6rem 1.6rem !important;
}
.stDownloadButton > button:hover { background: #EAF1FE !important; }

/* กล่องผลลัพธ์ 3 ระดับ */
.result-card {
    border-radius: 16px; padding: 1.4rem 1.6rem;
    box-shadow: 0 4px 14px rgba(20,51,124,0.07);
}
.res-high { background: #FDECEC; border: 1.5px solid #F5B5B5; }
.res-mid  { background: #FFF7E6; border: 1.5px solid #F5D98F; }
.res-low  { background: #EAF9F0; border: 1.5px solid #A9E3C0; }
.res-title { font-family: 'Prompt', sans-serif !important; font-size: 1.5rem; font-weight: 700; }
.res-high .res-title { color: #C0392B; }
.res-mid  .res-title { color: #B7791F; }
.res-low  .res-title { color: #1E8E4E; }
.res-note { font-size: 0.98rem; color: #33475F; margin-top: 4px; }
.res-prob {
    font-family: 'Prompt', sans-serif !important;
    font-size: 3rem; font-weight: 700; line-height: 1;
}
.res-high .res-prob { color: #C0392B; }
.res-mid  .res-prob { color: #B7791F; }
.res-low  .res-prob { color: #1E8E4E; }

/* อัปโหลดไฟล์ */
[data-testid="stFileUploaderDropzone"] {
    background: #FFFFFF !important;
    border: 2px dashed #9DBBEA !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploaderDropzone"] div, [data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small { color: #46608A !important; }
[data-testid="stFileUploaderDropzone"] button {
    background: #2563EB !important; color: #fff !important;
    border-radius: 10px !important; border: none !important;
}

hr { border-color: #E3EBF7; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load('credit_default_model.pkl')


try:
    artifact = load_model()
except FileNotFoundError:
    st.error("ไม่พบไฟล์ credit_default_model.pkl กรุณาวางไว้ในโฟลเดอร์เดียวกับ app.py")
    st.stop()

model = artifact['model']
FEATURES = artifact['feature_columns']
scaler = artifact.get('scaler')
use_scaled = artifact.get('use_scaled', False)     # รองรับไฟล์โมเดลเวอร์ชันเก่า
model_name = artifact.get('model_name', 'Random Forest')
results = artifact.get('results')


def predict_proba(df_input: pd.DataFrame) -> np.ndarray:
    """ทำนายโอกาสผิดนัด — scale ข้อมูลอัตโนมัติถ้าโมเดลต้องการ (เช่น SVM/Logistic)"""
    X_in = scaler.transform(df_input) if (use_scaled and scaler is not None) else df_input
    return model.predict_proba(X_in)[:, 1]


# ---------------- ส่วนหัวเว็บ ----------------
st.markdown(f"""
<div class="hero-wrap">
    <span class="hero-chip">🤖 โมเดลที่ใช้: {model_name} · ข้อมูลลูกค้า 30,000 ราย</span>
    <p class="hero-title">ระบบประเมินความเสี่ยง<br/>การผิดนัดชำระบัตรเครดิต</p>
    <p class="hero-sub">วิเคราะห์โอกาสที่ลูกค้าจะผิดนัดชำระในเดือนถัดไป
    จากประวัติการชำระเงินย้อนหลัง 6 เดือน วงเงิน และข้อมูลพื้นฐาน
    ด้วยเทคนิค Machine Learning</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# แถวสถิติ
acc = f1v = aucv = None
if results:
    try:
        acc = f"{results['Accuracy'][model_name]*100:.1f}%"
        f1v = f"{results['F1'][model_name]:.3f}"
        aucv = f"{results['AUC'][model_name]:.3f}"
    except (KeyError, TypeError):
        pass

c1, c2, c3, c4 = st.columns(4)
for col, num, label in [
    (c1, "30,000", "จำนวนลูกค้าที่ใช้เทรน"),
    (c2, acc or "78.8%", "ความแม่นยำ (Accuracy)"),
    (c3, aucv or "0.77", "คะแนน AUC"),
    (c4, "4", "โมเดลที่เปรียบเทียบ"),
]:
    col.markdown(f'<div class="stat-card"><div class="stat-num">{num}</div>'
                 f'<div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

# ตารางเปรียบเทียบโมเดล (ถ้ามีข้อมูลใน artifact)
if results:
    with st.expander("📊 ดูผลเปรียบเทียบทั้ง 4 โมเดล (Logistic / Random Forest / Gradient Boosting / SVM)"):
        st.dataframe(pd.DataFrame(results).round(4), use_container_width=True)
        st.caption(f"โมเดลที่นำมาใช้ทำนายในระบบนี้คือ {model_name} "
                   "(Support Vector Machine ที่ใช้ RBF kernel)")

st.markdown("<br/>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 ทำนายรายบุคคล", "📁 ทำนายจากไฟล์ CSV"])

# ---------------- แท็บ 1: รายบุคคล ----------------
with tab1:
    st.markdown('<p class="sec-title">กรอกข้อมูลลูกค้า</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown('<div class="group-head">👤 ข้อมูลส่วนตัว</div>', unsafe_allow_html=True)
        limit_bal = st.number_input("วงเงินบัตร (บาท)", 10000, 1000000, 100000, step=10000)
        sex = st.selectbox("เพศ", [1, 2], format_func=lambda x: "ชาย" if x == 1 else "หญิง")
        education = st.selectbox("ระดับการศึกษา", [1, 2, 3, 4],
                                 format_func=lambda x: {1: "ปริญญาโท/เอก", 2: "ปริญญาตรี",
                                                        3: "มัธยมศึกษา", 4: "อื่นๆ"}[x])
        marriage = st.selectbox("สถานภาพสมรส", [1, 2, 3],
                                format_func=lambda x: {1: "แต่งงานแล้ว", 2: "โสด", 3: "อื่นๆ"}[x])
        age = st.number_input("อายุ (ปี)", 21, 79, 35)

    pay_options = list(range(-2, 9))
    pay_label = lambda x: {-2: "ไม่มีการใช้จ่าย", -1: "จ่ายเต็มจำนวน",
                           0: "จ่ายขั้นต่ำตรงเวลา"}.get(x, f"ค้างชำระ {x} เดือน")

    with col2:
        st.markdown('<div class="group-head">📅 ประวัติการชำระ 6 เดือน</div>', unsafe_allow_html=True)
        pay_1 = st.selectbox("เดือนล่าสุด", pay_options, index=2, format_func=pay_label)
        pay_2 = st.selectbox("2 เดือนก่อน", pay_options, index=2, format_func=pay_label)
        pay_3 = st.selectbox("3 เดือนก่อน", pay_options, index=2, format_func=pay_label)
        pay_4 = st.selectbox("4 เดือนก่อน", pay_options, index=2, format_func=pay_label)
        pay_5 = st.selectbox("5 เดือนก่อน", pay_options, index=2, format_func=pay_label)
        pay_6 = st.selectbox("6 เดือนก่อน", pay_options, index=2, format_func=pay_label)

    with col3:
        st.markdown('<div class="group-head">💰 ยอดบิลและยอดชำระ (บาท)</div>', unsafe_allow_html=True)
        bill_amt1 = st.number_input("ยอดบิลเดือนล่าสุด", value=50000)
        bill_amt2 = st.number_input("ยอดบิล 2 เดือนก่อน", value=48000)
        bill_amt3 = st.number_input("ยอดบิล 3 เดือนก่อน", value=46000)
        bill_avg_rest = st.number_input("ยอดบิลเฉลี่ยเดือนที่ 4-6", value=45000)
        pay_amt1 = st.number_input("ยอดชำระเดือนล่าสุด", value=2000, min_value=0)
        pay_amt2 = st.number_input("ยอดชำระ 2 เดือนก่อน", value=2000, min_value=0)
        pay_avg_rest = st.number_input("ยอดชำระเฉลี่ยเดือนที่ 3-6", value=2000, min_value=0)

    st.markdown("<br/>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.2, 1])
    predict = mid.button("🎯 ประเมินความเสี่ยง", use_container_width=True)

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

        proba = float(predict_proba(input_data)[0])

        if proba >= 0.5:
            css, verdict, note = "res-high", "⚠️ ความเสี่ยงสูง", \
                "มีแนวโน้มผิดนัดชำระในเดือนถัดไป ควรพิจารณามาตรการติดตามหรือปรับเงื่อนไขวงเงิน"
        elif proba >= 0.3:
            css, verdict, note = "res-mid", "🟡 ความเสี่ยงปานกลาง", \
                "ควรติดตามพฤติกรรมการชำระอย่างใกล้ชิดในช่วง 1-2 เดือนข้างหน้า"
        else:
            css, verdict, note = "res-low", "✅ ความเสี่ยงต่ำ", \
                "ลูกค้ามีแนวโน้มชำระเงินตามปกติ"

        st.markdown("<br/>", unsafe_allow_html=True)
        r1, r2 = st.columns([1, 2], gap="large")
        with r1:
            st.markdown(f'<div class="result-card {css}" style="text-align:center">'
                        f'<div class="stat-label">โอกาสผิดนัดชำระ</div>'
                        f'<div class="res-prob">{proba:.1%}</div></div>',
                        unsafe_allow_html=True)
        with r2:
            st.markdown(f'<div class="result-card {css}">'
                        f'<div class="res-title">{verdict}</div>'
                        f'<div class="res-note">{note}</div></div>',
                        unsafe_allow_html=True)
            st.progress(proba)

# ---------------- แท็บ 2: ไฟล์ CSV ----------------
with tab2:
    st.markdown('<p class="sec-title">อัปโหลดไฟล์เพื่อทำนายหลายรายการ</p>', unsafe_allow_html=True)
    st.markdown("ไฟล์ CSV ต้องมีคอลัมน์เหมือนชุดข้อมูล UCI (มีหรือไม่มีคอลัมน์คำตอบก็ได้) "
                "ระบบจะทำนายทุกแถวและเรียงลำดับจากความเสี่ยงสูงไปต่ำ")

    uploaded = st.file_uploader("เลือกไฟล์ CSV", type="csv")

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
            with st.spinner(f"กำลังทำนาย {len(data):,} รายการด้วยโมเดล {model_name} "
                            "กรุณารอสักครู่..."):
                data['โอกาสผิดนัด'] = predict_proba(data[FEATURES]).round(4)
            data['ผลทำนาย'] = np.where(data['โอกาสผิดนัด'] >= 0.5, 'ผิดนัด', 'จ่ายปกติ')

            n_default = int((data['ผลทำนาย'] == 'ผิดนัด').sum())
            b1, b2, b3 = st.columns(3)
            for col, num, label in [
                (b1, f"{len(data):,}", "รายการทั้งหมด"),
                (b2, f"{n_default:,}", "คาดว่าผิดนัด"),
                (b3, f"{n_default/len(data):.1%}", "สัดส่วนความเสี่ยง"),
            ]:
                col.markdown(f'<div class="stat-card"><div class="stat-num">{num}</div>'
                             f'<div class="stat-label">{label}</div></div>',
                             unsafe_allow_html=True)

            st.markdown("<br/>", unsafe_allow_html=True)
            st.dataframe(data.sort_values('โอกาสผิดนัด', ascending=False),
                         use_container_width=True)

            csv = data.to_csv(index=False).encode('utf-8-sig')
            st.download_button("⬇️ ดาวน์โหลดผลลัพธ์ (CSV)", csv,
                               "prediction_results.csv", "text/csv")

st.markdown('<br/><p style="text-align:center;color:#8AA0BF;font-size:0.82rem">'
            'ผลการทำนายเป็นเครื่องมือช่วยประเมินความเสี่ยงเบื้องต้นเท่านั้น ไม่ใช่คำตัดสินทางการเงิน<br/>'
            'พัฒนาด้วย Streamlit · ชุดข้อมูล UCI Default of Credit Card Clients</p>',
            unsafe_allow_html=True)
