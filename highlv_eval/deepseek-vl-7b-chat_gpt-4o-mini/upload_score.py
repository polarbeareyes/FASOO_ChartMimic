import json
import re

# 원본 JSON 파일 경로
input_path = '/root/SOJUNG_STUFF/ChartMimic/highlv_eval/gemma-cosine_gpt-4o-mini/highlv_results.json'
# 수정된 JSON을 저장할 경로
output_path = '/root/SOJUNG_STUFF/ChartMimic/highlv_eval/gemma-cosine_gpt-4o-mini/highlv_results_uploaded_scores.json'

# 정규표현식: Score: **75/100**
score_pattern = re.compile(r"Score:\s*\*\*(\d+)/100\*\*")

# 파일 열기
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 각 항목 순회하며 score 수정
for item in data:
    comment = item.get('comment', '') or ''
    match = score_pattern.search(comment)
    if match:
        new_score = int(match.group(1))
        item['score'] = new_score

# 결과 저장
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"✅ Done. Updated scores saved to {output_path}")
