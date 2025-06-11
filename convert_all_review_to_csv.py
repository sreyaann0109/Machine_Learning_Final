import os
import re
import xml.etree.ElementTree as ET
import csv

ALLOWED_RATINGS = {1.0: 0, 2.0: 1, 4.0: 2, 5.0: 3}
ROOT_DIR = "datasets/sorted_data"
OUTPUT_CSV = "star_rating_dataset.csv"

def extract_reviews(xml_text):
    # 把每個 <review>...</review> 單獨切出
    pattern = re.compile(r"<review>.*?</review>", re.DOTALL)
    return pattern.findall(xml_text)

def parse_review_block(block):
    try:
        elem = ET.fromstring(block)
        rating_text = elem.findtext("rating")
        review_text = elem.findtext("review_text")

        if rating_text and review_text:
            rating = float(rating_text.strip())
            if rating in ALLOWED_RATINGS:
                label = ALLOWED_RATINGS[rating]
                text = review_text.strip().replace("\n", " ")
                return text, label
    except:
        return None
    return None

all_texts, all_labels = [], []
for folder in os.listdir(ROOT_DIR):
    subdir = os.path.join(ROOT_DIR, folder)
    review_path = os.path.join(subdir, "all.review")
    if os.path.isfile(review_path):
        print(f"🔍 處理 {review_path}...")
        try:
            with open(review_path, "r", encoding="utf-8", errors="ignore") as f:
                xml_content = f.read()
            blocks = extract_reviews(xml_content)
            count = 0
            for blk in blocks:
                result = parse_review_block(blk)
                if result:
                    text, label = result
                    all_texts.append(text)
                    all_labels.append(label)
                    count += 1
            print(f"✅ 取得 {count} 筆")
        except Exception as e:
            print(f"❌ 無法處理 {review_path}：{e}")

# 寫入 CSV
print(f"📄 寫入 {OUTPUT_CSV}...")
with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    for text, label in zip(all_texts, all_labels):
        writer.writerow([text, label])

print(f"✅ 共儲存 {len(all_texts)} 筆資料到 {OUTPUT_CSV}")