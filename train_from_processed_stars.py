import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_bow_review_file(filepath):
    X, y = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            feature_dict = {}
            label = None
            for item in line:
                if item.startswith('#label#'):
                    label = float(item.split(':')[1])
                else:
                    word, count = item.rsplit(':', 1)
                    feature_dict[word] = int(count)
            if label is not None:
                X.append(feature_dict)
                y.append(label)
    return X, y

def load_all_domains(base_path, domains):
    all_X, all_y = [], []
    for domain in domains:
        path = os.path.join(base_path, domain, 'all_balanced.review')
        X, y = load_bow_review_file(path)
        all_X.extend(X)
        all_y.extend(y)
        print(f"✅ 讀取 {domain}, 筆數：{len(X)}")
    return all_X, all_y


# 1. 載入多個領域資料
domains = ["books", "dvd", "electronics", "kitchen"]
X_raw, y = load_all_domains("datasets/processed_stars", domains)

# 2. 向量化與切分
vec = DictVectorizer()
X = vec.fit_transform(X_raw)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 模型訓練
model = LogisticRegression(solver="saga", max_iter=1000, n_jobs=-1, class_weight="balanced")
model.fit(X_train, y_train)

# 4. 評估
y_pred = model.predict(X_test)
print("📊 分類報告：")
print(classification_report(y_test, y_pred))

# 5. 儲存
import joblib
joblib.dump(model, "multi_domain_model.pkl")
joblib.dump(vec, "multi_domain_vectorizer.pkl")