import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import joblib
import torch
from transformers import BertModel
from sklearn.utils import shuffle

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # 可有可無，但保險起見保留

    # ✅ 顯示 CUDA 狀態
    print("CUDA 是否可用:", torch.cuda.is_available())
    print("使用裝置:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # 測試 BERT 是否能上 GPU（可刪）
    model_test = BertModel.from_pretrained("bert-base-uncased")
    model_test.to("cuda" if torch.cuda.is_available() else "cpu")
    print("模型所在裝置:", next(model_test.parameters()).device)

    # ✅ 1. 載入 CSV，均衡抽樣
    df = pd.read_csv("star_rating_dataset.csv")  # 應包含 text 與 label 欄位
    label_col = "label"
    unique_labels = sorted(df[label_col].unique())
    # ✅ 印出每個星等有多少筆資料
    print("📊 每個星等的資料數量：")
    print(df["label"].value_counts().sort_index())

    sample_per_class = 2000  # ✅ 每個類別最多抽幾筆
    balanced_df = (
        df.groupby(label_col)
        .apply(lambda x: x.sample(n=min(len(x), sample_per_class), random_state=42))
        .reset_index(drop=True)
    )

    # ✅ 打亂 + 分割 train/test（8:2）
    balanced_df = shuffle(balanced_df, random_state=42)
    dataset = Dataset.from_pandas(balanced_df)
    dataset = dataset.train_test_split(test_size=0.2)

    # ✅ 2. Tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        texts = batch["text"]
        texts = [t if isinstance(t, str) else "" for t in texts]
        return tokenizer(texts, truncation=True, padding="max_length", max_length=256)

    tokenized = dataset.map(tokenize, batched=True)

    # ✅ 3. 模型初始化
    num_labels = len(unique_labels)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # ✅ 4. 訓練參數設定
    args = TrainingArguments(
        output_dir="./bert_star_rating",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        dataloader_num_workers=3,
        dataloader_pin_memory=True,
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        metric_for_best_model="accuracy"
    )

    # ✅ 5. 評估指標
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        acc = (preds == labels).mean()
        return {"accuracy": acc}

    # ✅ 6. 開始訓練
    print("🧠 開始訓練 BERT 模型...")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()

    # ✅ 7. 儲存模型與 tokenizer
    trainer.save_model("bert_star_rating_model")
    tokenizer.save_pretrained("bert_star_rating_model")

    # ✅ 8. 儲存 label map（預測時要用）
    label_map = {i: label for i, label in enumerate(unique_labels)}
    joblib.dump(label_map, "bert_label_map.pkl")
