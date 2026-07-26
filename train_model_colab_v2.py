# ============================================================
# ทำนายการผิดนัดชำระบัตรเครดิต (Default of Credit Card Clients)
# เวอร์ชัน 2: เพิ่ม Support Vector Machine (SVM)
# สำหรับรันใน Google Colab — แบ่งรันทีละ STEP
# ============================================================

# ---------- STEP 0: อัปโหลดไฟล์ข้อมูล (รันใน Colab) ----------
# from google.colab import files
# uploaded = files.upload()   # เลือกไฟล์ UCI_Credit_Card.csv

# ---------- STEP 1: Import Library ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             classification_report, confusion_matrix,
                             RocCurveDisplay)

# ---------- STEP 2: โหลดข้อมูล ----------
df = pd.read_csv('UCI_Credit_Card.csv')
print("ขนาดข้อมูล:", df.shape)
df.head()

# ---------- STEP 3: สำรวจข้อมูล (EDA) ----------
print(df.info())
print("\nค่าว่าง:", df.isnull().sum().sum(), "ค่า")
print("\nสัดส่วน target (0=จ่ายปกติ, 1=ผิดนัด):")
print(df['default.payment.next.month'].value_counts(normalize=True))

plt.figure(figsize=(5, 4))
df['default.payment.next.month'].value_counts().plot(kind='bar', color=['#4C9F70', '#D9534F'])
plt.title('Default (1) vs No Default (0)')
plt.xticks(rotation=0)
plt.show()

# ---------- STEP 4: ทำความสะอาดข้อมูล ----------
df = df.rename(columns={'default.payment.next.month': 'default', 'PAY_0': 'PAY_1'})
df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})   # ค่านอกนิยาม -> อื่นๆ
df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})                 # ค่านอกนิยาม -> อื่นๆ

X = df.drop(columns=['ID', 'default'])
y = df['default']
FEATURE_COLUMNS = X.columns.tolist()

# ---------- STEP 5: แบ่ง Train / Test + Scaling ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Train:", X_train.shape, "| Test:", X_test.shape)

# ---------- STEP 6: เทรนและเปรียบเทียบ 4 โมเดล ----------
# หมายเหตุ: SVM ใช้เวลาเทรนนานสุด (~3-8 นาทีใน Colab) เพราะข้อมูล 24,000 แถว
# และเปิด probability=True เพื่อให้คำนวณ predict_proba ได้
# SVM จำเป็นต้องใช้ข้อมูลที่ scale แล้ว (use_scaled=True) เพราะคำนวณจากระยะทาง
models = {
    'Logistic Regression': (LogisticRegression(max_iter=1000, class_weight='balanced'), True),
    'Random Forest': (RandomForestClassifier(n_estimators=200, max_depth=10,
                                             class_weight='balanced',
                                             random_state=42, n_jobs=-1), False),
    'Gradient Boosting': (GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                     random_state=42), False),
    'SVM (RBF kernel)': (SVC(kernel='rbf', C=1.0, class_weight='balanced',
                             probability=True, cache_size=500,
                             random_state=42), True),
}

results = {}
for name, (model, use_scaled) in models.items():
    print(f"กำลังเทรน {name} ...")
    Xtr = X_train_scaled if use_scaled else X_train
    Xte = X_test_scaled if use_scaled else X_test
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    results[name] = {
        'Accuracy': accuracy_score(y_test, pred),
        'F1': f1_score(y_test, pred),
        'AUC': roc_auc_score(y_test, proba),
    }

results_df = pd.DataFrame(results).T.round(4)
print("\nผลเปรียบเทียบ 4 โมเดล:")
print(results_df)

# กราฟเปรียบเทียบ
results_df.plot(kind='bar', figsize=(9, 5), rot=15)
plt.title('เปรียบเทียบประสิทธิภาพ 4 โมเดล')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# ---------- STEP 7: เลือกโมเดลที่จะนำไปใช้งานจริง ----------
# กำหนดให้ใช้ SVM ตามโจทย์ที่อาจารย์กำหนด
# (ถ้าอยากเปลี่ยนเป็นโมเดลอื่น แก้ค่าตัวแปร FINAL_MODEL บรรทัดล่างนี้)
FINAL_MODEL = 'SVM (RBF kernel)'

best_name = FINAL_MODEL
best_model, best_scaled = models[best_name]
print(f"โมเดลที่เลือกใช้งาน: {best_name}")
print(f"(โมเดลที่ AUC สูงสุดในการทดลองนี้คือ {results_df['AUC'].idxmax()} "
      f"แต่เลือกใช้ {best_name} ตามข้อกำหนดของงาน)")
print(f"จำนวน Support Vectors: {best_model.n_support_.sum():,}")

Xte_best = X_test_scaled if best_scaled else X_test
y_pred = best_model.predict(Xte_best)

print(classification_report(y_test, y_pred,
      target_names=['จ่ายปกติ (0)', 'ผิดนัด (1)']))

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['จ่ายปกติ', 'ผิดนัด'], yticklabels=['จ่ายปกติ', 'ผิดนัด'])
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.title(f'Confusion Matrix - {best_name}')
plt.show()

RocCurveDisplay.from_estimator(best_model, Xte_best, y_test)
plt.title(f'ROC Curve - {best_name}')
plt.show()

# ---------- STEP 8: บันทึกโมเดลไปใช้กับ Streamlit ----------
# บันทึก: โมเดลที่ดีที่สุด + scaler + ธง use_scaled + รายชื่อคอลัมน์
artifact = {
    'model': best_model,
    'model_name': best_name,
    'scaler': scaler,
    'use_scaled': best_scaled,      # ถ้า True แอปต้อง scale ข้อมูลก่อนทำนาย
    'feature_columns': FEATURE_COLUMNS,
    'results': results_df.to_dict(),  # เก็บผลเปรียบเทียบไว้โชว์ในเว็บ
}
joblib.dump(artifact, 'credit_default_model.pkl', compress=3)
print("บันทึกโมเดลเรียบร้อย -> credit_default_model.pkl")

# ตรวจสอบว่าบันทึกเป็นโมเดลที่ต้องการจริงหรือไม่
check = joblib.load('credit_default_model.pkl')
print("ชนิดโมเดลในไฟล์:", type(check['model']).__name__,
      "| ชื่อ:", check['model_name'])

# from google.colab import files
# files.download('credit_default_model.pkl')

# ---------- STEP 9: ทดสอบโหลดโมเดลกลับมาใช้ ----------
loaded = joblib.load('credit_default_model.pkl')
sample = X_test.iloc[[0]]
sample_input = loaded['scaler'].transform(sample) if loaded['use_scaled'] else sample
prob = loaded['model'].predict_proba(sample_input)[0, 1]
print(f"ตัวอย่างการทำนาย ({loaded['model_name']}): โอกาสผิดนัด = {prob:.2%}")
