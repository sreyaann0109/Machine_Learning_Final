from transformers import pipeline
import joblib

def main():
    # 1. 載入模型與 tokenizer
    model_dir = "bert_star_rating_model"
    classifier = pipeline("text-classification", model=model_dir, tokenizer=model_dir)

    # 2. 載入標籤對應表（label → 星等）
    label_map = {
        0: 1.0,
        1: 2.0,
        2: 4.0,
        3: 5.0
    }

    # 3. 輸入評論文字
    text = input("請輸入評論文字：\n")

    # 4. 使用 pipeline 預測
    result = classifier(text)[0]
    label_id = int(result["label"].split("_")[-1])  # 轉成整數 label，比如 LABEL_2 → 2
    star_rating = label_map.get(label_id, "未知")

    # 5. 印出結果
    print(f"\n⭐️ 預測星等：{star_rating}（置信度 {result['score']:.3f}）")

if __name__ == "__main__":
    main()