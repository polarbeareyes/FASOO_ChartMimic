import json
import os

input_path = "/root/SOJUNG_STUFF/ChartMimic/highlv_eval/gemma-trained-direct_checker/highlv_results.json"
output_path = "/root/SOJUNG_STUFF/ChartMimic/output_converted_gemma-trained-direct_checker.jsonl"  # 확장자 .jsonl 로 바꾸는 게 관례

base_org_dir = "/root/SOJUNG_STUFF/ChartMimic/dataset/direct_600/"
base_gen_dir = "/root/SOJUNG_STUFF/ChartMimic/results/direct/chart2code_gemma-trained-direct_checker_DirectAgent_results/direct_checker"

with open(input_path, "r", encoding="utf-8") as f_in:
    data = json.load(f_in)

with open(output_path, "w", encoding="utf-8") as f_out:
    for entry in data:
        file_name = entry.get("file")
        score = entry.get("score", 0)

        original_path_dir = os.path.join(base_org_dir, file_name)
        generated_path_dir = os.path.join(base_gen_dir, file_name)

        original_path = original_path_dir+ ".py"
        generated_path = generated_path_dir+ ".py"
        reason =  entry.get("reason")
        new_entry = {
            "orginial": original_path,
            "generated": generated_path,
            "gpt4v_score": score,
            "reason": reason
        }

        json_line = json.dumps(new_entry, ensure_ascii=False)
        f_out.write(json_line + "\n")
 