import os
import joblib
from collections import Counter, defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import numpy as np

INPUT_BASE = "datasets/sorted_data"
INPUT_FILENAME = "all.bow.review"

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

if __name__ == "__main__":
    # 載入資料
    X_raw, y = load_all_domains(INPUT_BASE)

    # 類別對應與編碼
    class_counts = Counter(y)
    print("📊 類別數量：", class_counts)

    unique_classes = sorted(class_counts.keys())  # ex: [1.0, 2.0, 4.0, 5.0]
    class_map = {c: i for i, c in enumerate(unique_classes)}       # 1.0 → 0, 2.0 → 1...
    class_inv_map = {i: c for c, i in class_map.items()}           # 0 → 1.0 ...
    joblib.dump(class_inv_map, "class_inv_map.pkl")                # 儲存供預測用

    y_mapped = np.array([class_map[label] for label in y])         # 轉為整數標籤

    # 建立 sample_weight
    max_count = max(class_counts.values())
    class_weights = {class_map[k]: max_count / v for k, v in class_counts.items()}
    sample_weight_full = np.array([class_weights[label] for label in y_mapped])

    print("📘 類別權重：", class_weights)

    # 向量化
    print("🔧 向量化中...")
    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(X_raw)

    # 分割訓練測試
    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
        X, y_mapped, sample_weight_full, test_size=0.2, random_state=42
    )

    # 模型訓練
    print("🧠 訓練模型中...")
    model = XGBClassifier(
        objective='multi:softmax',
        num_class=len(unique_classes),
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train, sample_weight=sw_train)

    # 預測與評估
    y_pred = model.predict(X_test)
    y_test_real = [class_inv_map[int(y)] for y in y_test]
    y_pred_real = [class_inv_map[int(p)] for p in y_pred]

    print("📊 模型評估結果：")
    print(classification_report(y_test_real, y_pred_real, digits=3))

    # 儲存模型與向量器
    joblib.dump(model, "star_rating_weight_model.pkl")
    joblib.dump(vectorizer, "star_rating_weight_vectorizer.pkl")
    print("💾 模型與向量器已儲存")