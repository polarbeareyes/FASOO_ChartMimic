import json

input_path = "/root/SOJUNG_STUFF/ChartMimic/results/direct/chart2code_deepseek-vl-7b-chat_DirectAgent_results_code4evaluation.json"

# F1 점수 저장 리스트
text_f1_list = []
chart_type_f1_list = []
layout_f1_list = []
color_f1_list = []

# 전체 JSON 파일 로드 (한 번에)
with open(input_path, "r") as f:
    all_data = json.load(f)  # <== 바로 리스트 전체 읽기

# 각 항목에서 F1 점수 추출
for data in all_data:
    text_f1 = data.get("text_metrics", {}).get("f1")
    chart_type_f1 = data.get("chart_type_metrics", {}).get("f1")
    layout_f1 = data.get("layout_metrics", {}).get("f1")
    color_f1 = data.get("color_metrics", {}).get("f1")

    if text_f1 is not None:
        text_f1_list.append(text_f1)
    if chart_type_f1 is not None:
        chart_type_f1_list.append(chart_type_f1)
    if layout_f1 is not None:
        layout_f1_list.append(layout_f1)
    if color_f1 is not None:
        color_f1_list.append(color_f1)

# 평균 계산 함수
def safe_average(lst):
    print(len(lst))
    return sum(lst) / 600 

# 결과 정리
result = {
    "average_text_f1": safe_average(text_f1_list),
    "average_chart_type_f1": safe_average(chart_type_f1_list),
    "average_layout_f1": safe_average(layout_f1_list),
    "average_color_f1": safe_average(color_f1_list)
}


# 결과 저장
with open("f1_averages.json", "w") as f:
    json.dump(result, f, indent=2)

# 출력 확인
print(json.dumps(result, indent=2))
print(f"\n✅ text_metrics가 존재하는 항목 수: {len(text_f1_list)}개")

