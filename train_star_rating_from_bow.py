import os
import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
import numpy as np
from collections import defaultdict
import random

INPUT_BASE = "datasets/sorted_data"
INPUT_FILENAME = "all.bow.review"
MAX_SAMPLES = None  # ✅ 最多使用多少筆資料，設為 None 則使用全部

def load_bow_review_file(filepath):
    X, y = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            bow = {}
            label = None
            for item in parts:
                if item.startswith("#label#"):
                    label = float(item.split(":")[1])
                else:
                    word, count = item.rsplit(":", 1)
                    bow[word] = int(count)
            if label is not None:
                X.append(bow)
                y.append(label)
    return X, y

# 1. 載入所有分類的 .bow.review
def load_all_domains(base_path):
    all_X, all_y = [], []
    for domain in os.listdir(base_path):
        bow_path = os.path.join(base_path, domain, INPUT_FILENAME)
        if os.path.isfile(bow_path):
            print(f"📂 讀取 {bow_path}...")
            X, y = load_bow_review_file(bow_path)
            all_X.extend(X)
            all_y.extend(y)
            print(f"  ✅ 筆數：{len(X)}")
    return all_X, all_y

def balance_dataset(X_raw, y):
    grouped = defaultdict(list)
    for x, label in zip(X_raw, y):
        grouped[label].append(x)

    min_class_size = min(len(g) for g in grouped.values())  # 例：最少的星等有 16,347 筆

    X_balanced, y_balanced = [], []
    for label, samples in grouped.items():
        sampled = random.sample(samples, min_class_size)
        X_balanced.extend(sampled)
        y_balanced.extend([label] * min_class_size)

    print(f"⚖️ 平衡後每類別筆數：{min_class_size}，總數：{len(X_balanced)}")
    return X_balanced, y_balanced

if __name__ == "__main__":
    # 讀取資料
    X_raw, y = load_all_domains(INPUT_BASE)
    X_raw, y = balance_dataset(X_raw, y)

    unique_classes = sorted(set(y))  # [1.0, 2.0, 4.0, 5.0]
    class_map = {c: i for i, c in enumerate(unique_classes)}
    class_inv_map = {i: c for c, i in class_map.items()}
    print("📘 類別對應：", class_map)

    # 儲存還原對應表，供預測用
    joblib.dump(class_inv_map, "class_inv_map.pkl")
    # 轉換成連續整數標籤
    y_mapped = np.array([class_map[label] for label in y])

    # ✅ 限制樣本數量（加速訓練）
    if MAX_SAMPLES and len(X_raw) > MAX_SAMPLES:
        X_raw = X_raw[:MAX_SAMPLES]
        y = y[:MAX_SAMPLES]
        print(f"⚡ 使用前 {MAX_SAMPLES} 筆資料訓練")

    # 向量化
    print("🔧 向量化中...")
    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(X_raw)

    # 切分訓練/測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y_mapped, test_size=0.2, random_state=42)

    # 建立並訓練模型
    print("🧠 訓練模型中...")
    # model = LogisticRegression(solver="saga", max_iter=1000, class_weight="balanced", n_jobs=-1)
    # model = SGDClassifier(loss="log_loss", max_iter=1000, class_weight="balanced", n_jobs=-1, random_state=42)
    model = XGBClassifier(
        objective='multi:softmax',
        num_class=5,
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 預測與評估
    print("📊 模型評估結果：")
    y_pred = model.predict(X_test)
    y_pred_real = [class_inv_map[int(p)] for p in y_pred]
    y_test_real = [class_inv_map[int(t)] for t in y_test]
    print(classification_report(y_test_real, y_pred_real, digits=3))

    # 儲存模型與向量器
    joblib.dump(model, "star_rating_model.pkl")
    joblib.dump(vectorizer, "star_rating_vectorizer.pkl")
    print("💾 模型與向量器已儲存")