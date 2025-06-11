import joblib
import re
from collections import Counter
import numpy as np

# 🔧 請根據你要用的模型路徑調整：
MODEL_PATH = "star_rating_weight_model.pkl"
VEC_PATH = "star_rating_weight_vectorizer.pkl"

# 試著載入類別還原對應表（如果有）
try:
    CLASS_INV_MAP = joblib.load("class_inv_map.pkl")  # {0:1.0, 1:2.0, 2:4.0, 3:5.0}
except:
    CLASS_INV_MAP = None

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    tokens = text.split()
    return dict(Counter(tokens))

# 載入模型與向量器
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

# 輸入文字
text = input("請輸入評論文字：\n")
features = preprocess_text(text)
X_vec = vectorizer.transform([features])

# 預測
pred = model.predict(X_vec)[0]

# 還原星等（若模型有編碼）
if CLASS_INV_MAP and int(pred) in CLASS_INV_MAP:
    real_star = CLASS_INV_MAP[int(pred)]
else:
    real_star = pred  # 原始星等模型，直接輸出

print(f"\n⭐️ 預測星等：{real_star}")