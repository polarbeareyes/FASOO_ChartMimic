
import json
with open("/root/SOJUNG_STUFF/ChartMimic/highlv_eval/deepseek-vl-7b-chat_gpt-4o/highlv_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)
# 전체 score 합과 개수 계산
total_score = sum(entry["score"] for entry in data)
score_count = len([entry for entry in data if "score" in entry])

print(f"총 Score 합계: {total_score/600}")
print(f"Score 개수: {score_count}")