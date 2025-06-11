import os
import re
import xml.etree.ElementTree as ET
from collections import Counter

INPUT_BASE = "datasets/sorted_data"
OUTPUT_FILENAME = "all.bow.review"

def text_to_bow(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    tokens = text.split()
    return dict(Counter(tokens))

def parse_review_block(review_str):
    try:
        review_str = review_str.replace("&", "&amp;")
        elem = ET.fromstring(review_str)
        rating_elem = elem.find("rating")
        text_elem = elem.find("review_text")
        if rating_elem is None or text_elem is None:
            return None
        rating = float(rating_elem.text.strip())
        if rating == 3.0:
            return None
        text = text_elem.text or ""
        bow = text_to_bow(text)
        line = " ".join(f"{w}:{c}" for w, c in bow.items()) + f" #label#:{rating}"
        return line
    except:
        return None

def parse_all_review_by_block(input_path):
    with open(input_path, 'r', encoding='latin1') as f:
        lines = []
        inside = False
        review_lines = []

        for line in f:
            if '<review>' in line:
                inside = True
                review_lines = [line]
            elif '</review>' in line:
                review_lines.append(line)
                inside = False
                review_xml = ''.join(review_lines)
                parsed = parse_review_block(review_xml)
                if parsed:
                    lines.append(parsed)
            elif inside:
                review_lines.append(line)

        return lines

# 主程式：逐分類處理
if __name__ == "__main__":
    for domain in os.listdir(INPUT_BASE):
        domain_path = os.path.join(INPUT_BASE, domain)
        input_file = os.path.join(domain_path, "all.review")
        output_file = os.path.join(domain_path, OUTPUT_FILENAME)

        if not os.path.isfile(input_file):
            continue

        print(f"🔍 處理 {input_file}...")
        try:
            lines = parse_all_review_by_block(input_file)
            if lines:
                with open(output_file, 'w', encoding='utf-8') as f:
                    for line in lines:
                        f.write(line + "\n")
                print(f"✅ 輸出 {len(lines)} 筆 → {output_file}")
            else:
                print(f"⚠️ 無有效資料產生於 {input_file}")
        except Exception as e:
            print(f"❌ 錯誤於 {input_file}: {e}")