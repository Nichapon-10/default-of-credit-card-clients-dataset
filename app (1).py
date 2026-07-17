# ============================================================
# Streamlit App: ทำนายการผิดนัดชำระบัตรเครดิต
# วิธีรัน:  streamlit run app.py
# ต้องมีไฟล์ credit_default_model.pkl (ได้จากโค้ด Colab) อยู่โฟลเดอร์เดียวกัน
# ติดตั้ง: pip install streamlit scikit-learn joblib pandas numpy
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="ทำนายการผิดนัดชำระบัตรเครดิต",
                   page_icon="💳", layout="wide")


@st.cache_resource
def load_model():
    artifact = joblib.load('credit_default_model.pkl')
    return artifact


try:
    artifact = load_model()
    model = artifact['model']
    FEATURES = artifact['feature_columns']
except FileNotFoundError:
    st.error("ไม่พบไฟล์ credit_default_model.pkl กรุณาวางไฟล์โมเดลไว้โฟลเดอร์เดียวกับ app.py")
    st.stop()

st.title("💳 ทำนายการผิดนัดชำระบัตรเครดิต")
st.caption("โมเดล Random Forest เทรนจากชุดข้อมูล UCI Default of Credit Card Clients (ลูกค้า 30,000 ราย)")

tab1, tab2 = st.tabs(["🔍 ทำนายรายคน", "📁 ทำนายจากไฟล์ CSV"])

# ---------------- Tab 1: กรอกข้อมูลรายคน ----------------
with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("ข้อมูลส่วนตัว")
        limit_bal = st.number_input("วงเงินบัตร (LIMIT_BAL)", 10000, 1000000, 100000, step=10000)
        sex = st.selectbox("เพศ", options=[1, 2],
                           format_func=lambda x: "ชาย" if x == 1 else "หญิง")
        education = st.selectbox("การศึกษา", options=[1, 2, 3, 4],
                                 format_func=lambda x: {1: "ปริญญาโท/เอก", 2: "ปริญญาตรี",
                                                        3: "มัธยม", 4: "อื่นๆ"}[x])
        marriage = st.selectbox("สถานภาพ", options=[1, 2, 3],
                                format_func=lambda x: {1: "แต่งงาน", 2: "โสด", 3: "อื่นๆ"}[x])
        age = st.number_input("อายุ", 21, 79, 35)

    pay_options = list(range(-2, 9))
    pay_label = lambda x: {-2: "-2 ไม่มีการใช้จ่าย", -1: "-1 จ่ายเต็ม", 0: "0 จ่ายขั้นต่ำ"}.get(x, f"{x} ค้างจ่าย {x} เดือน")

    with col2:
        st.subheader("สถานะการชำระ 6 เดือน")
        pay_1 = st.selectbox("เดือนล่าสุด (PAY_1)", pay_options, index=2, format_func=pay_label)
        pay_2 = st.selectbox("2 เดือนก่อน (PAY_2)", pay_options, index=2, format_func=pay_label)
        pay_3 = st.selectbox("3 เดือนก่อน (PAY_3)", pay_options, index=2, format_func=pay_label)
        pay_4 = st.selectbox("4 เดือนก่อน (PAY_4)", pay_options, index=2, format_func=pay_label)
        pay_5 = st.selectbox("5 เดือนก่อน (PAY_5)", pay_options, index=2, format_func=pay_label)
        pay_6 = st.selectbox("6 เดือนก่อน (PAY_6)", pay_options, index=2, format_func=pay_label)

    with col3:
        st.subheader("ยอดบิลและยอดชำระ")
        bill_amt1 = st.number_input("ยอดบิลเดือนล่าสุด (BILL_AMT1)", value=50000)
        bill_amt2 = st.number_input("ยอดบิล 2 เดือนก่อน (BILL_AMT2)", value=48000)
        bill_amt3 = st.number_input("ยอดบิล 3 เดือนก่อน (BILL_AMT3)", value=46000)
        bill_avg_rest = st.number_input("ยอดบิลเฉลี่ยเดือน 4-6", value=45000,
                                        help="ใช้ค่าเดียวกันสำหรับ BILL_AMT4-6 เพื่อลดช่องกรอก")
        pay_amt1 = st.number_input("ยอดชำระเดือนล่าสุด (PAY_AMT1)", value=2000, min_value=0)
        pay_amt2 = st.number_input("ยอดชำระ 2 เดือนก่อน (PAY_AMT2)", value=2000, min_value=0)
        pay_avg_rest = st.number_input("ยอดชำระเฉลี่ยเดือน 3-6", value=2000, min_value=0)

    if st.button("🎯 ทำนาย", type="primary", use_container_width=True):
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

        proba = model.predict_proba(input_data)[0, 1]

        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("โอกาสผิดนัดชำระ", f"{proba:.1%}")
        with c2:
            if proba >= 0.5:
                st.error("⚠️ ความเสี่ยงสูง — มีแนวโน้มผิดนัดชำระเดือนถัดไป")
            elif proba >= 0.3:
                st.warning("🟡 ความเสี่ยงปานกลาง — ควรติดตามอย่างใกล้ชิด")
            else:
                st.success("✅ ความเสี่ยงต่ำ — มีแนวโน้มชำระตามปกติ")
            st.progress(float(proba))

# ---------------- Tab 2: อัปโหลด CSV ทำนายทีละหลายคน ----------------
with tab2:
    st.write("อัปโหลดไฟล์ CSV ที่มีคอลัมน์เหมือนชุดข้อมูล UCI (มีหรือไม่มีคอลัมน์ target ก็ได้)")
    uploaded = st.file_uploader("เลือกไฟล์ CSV", type="csv")

    if uploaded is not None:
        data = pd.read_csv(uploaded)
        # จัดการชื่อคอลัมน์ให้ตรงกับตอนเทรน
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

            st.success(f"ทำนายเสร็จ {len(data):,} รายการ | "
                       f"คาดว่าผิดนัด {(data['ผลทำนาย'] == 'ผิดนัด').sum():,} ราย "
                       f"({(data['ผลทำนาย'] == 'ผิดนัด').mean():.1%})")
            st.dataframe(data.sort_values('โอกาสผิดนัด', ascending=False),
                         use_container_width=True)

            csv = data.to_csv(index=False).encode('utf-8-sig')
            st.download_button("⬇️ ดาวน์โหลดผลลัพธ์", csv,
                               "prediction_results.csv", "text/csv")

st.divider()
st.caption("หมายเหตุ: ผลทำนายเป็นเพียงเครื่องมือช่วยประเมินความเสี่ยง ไม่ใช่คำตัดสินทางการเงิน")
