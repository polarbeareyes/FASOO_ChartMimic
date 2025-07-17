import re
import pandas as pd
import json
import os
from tqdm import tqdm
from argparse import ArgumentParser
from dotenv import load_dotenv
import json
from multiprocessing import Process
import glob

load_dotenv()


def extract_code(text):
    """Extract code from markdown text."""
    code = re.findall(r"```python(.*?)```", text, re.DOTALL)
    if len(code) == 0:
        code = [""]
    return code

def get_variable_code(file):
    edit_ori_file = "{}/dataset/customized_500/".format(os.environ["PROJECT_PATH"]) + file
    with open(edit_ori_file, "r") as f:
        code = f.read()
        pattern = re.compile(r"# ===================\n# Part 2: Data Preparation\n# ===================\n(.*?)# ===================\n# Part 3: Plot Configuration and Rendering\n# ===================", re.DOTALL)
        match = pattern.search(code)

        if match:
            extracted_text = match.group(1)
            extracted_text = extracted_text.strip() 
            extracted_text = "#Variable Code Block\nimport warnings;warnings.filterwarnings('ignore', category=UserWarning);warnings.filterwarnings('ignore', category=FutureWarning);import matplotlib.pyplot as plt;import pandas as pd;import numpy as np;np.random.seed(0);import math;from matplotlib_venn import venn2;from matplotlib import cm;from scipy.stats import gaussian_kde;import networkx as nx;from matplotlib.gridspec import GridSpec;from scipy.stats import multivariate_normal;import colorsys;import matplotlib.colors as mcolors;from matplotlib.colors import LogNorm;from scipy.stats import norm;import matplotlib.gridspec as gridspec;import seaborn as sns\n" + extracted_text
        else:
            print(edit_ori_file)
            raise ValueError("No match found")
    return extracted_text

def _muti_process_run(rank, data, num_processes):
    skip_stats = {
    "json_parse_error": 0,
    "no_code_block": 0,
    "part2_not_found": 0,
    "other_exception": 0
    }
    skipped_files = []


    sub_index = [_ for _ in range(len(data))][
        rank :: num_processes
    ]

    for i in tqdm(sub_index, disable=rank != 0):
        output_file = os.path.basename(data["file"][i]).replace(".pdf", ".py")
        output_file = output_dir + "/" + output_file

        try: 
            if "gpt" in input_file:
                try:
                    code = json.loads(data["response"][i])["choices"][0]["message"]["content"]
                    raw_model_response = json.loads(data["response"][i])["choices"][0]["message"]["content"]
                except:
                    code = ""
            elif "claude" in input_file:
                try:
                    code = data["response"][i]["choices"][0]["message"]["content"]
                    raw_model_response = data["response"][i]["choices"][0]["message"]["content"]
                except:
                    code = ""
            else:
                code = data["response"][i] if data["response"][i] else ""
                raw_model_response = data["response"][i]

        except Exception as e:
            print(f"[SKIP] ({output_file}) Failed to parse JSON response: {e}")
            skip_stats["json_parse_error"] += 1
            skipped_files.append({
                "file": output_file,
                "reason": "json_parse_error",
                "model_response": str(raw_model_response)
            })
            continue
        
        try: 
            if "idefics2" in input_file:
                if "```python" in code:
                    code = extract_code(code)[0]
                else:
                    code = code.split("Assistant: ")[1] 
            else:
                code = extract_code(code)[0]

            if code.strip() == "":
                print(f"[SKIP] ({output_file}) No code block extracted")
                skip_stats["no_code_block"] += 1
                skipped_files.append({
                    "file": output_file,
                    "reason": "no_code_block",
                    "model_response": str(raw_model_response)
                })
                continue

            if code == "":
                # No Code Found
                continue
    
            if "chartedit" in output_file.lower():
                try: 
                    variable_code = get_variable_code( os.path.basename(output_file) )
                    code = variable_code + "\n" + code
                except ValueError as e:
                        print(f"[SKIP] ({output_file}) Failed to extract Part 2 block (Customized): {e}")
                        skip_stats["part2_not_found"] += 1
                        skipped_files.append({
                            "file": output_file,
                            "reason": "part2_not_found",
                            "model_response": str(raw_model_response)
                        })
                        continue
             
            code = re.sub(r"plt\.savefig\(.*\n*", "", code, flags=re.S)
            code = re.sub(r"plt.show\(.*\n*", "", code, flags=re.S)
            code = (
                code.strip()
                + '\nplt.savefig("{}")'.format(
                    output_file.replace(".py", f".pdf")
                )
            )
    
            with open(output_file, "w") as f:
                f.write(code)
            # print(output_file)
            if "llava-v1.6-mistral-7b-hf_EditAgent_results/edit_checker/HR_11.py" not in output_file and "llava-v1.6-vicuna-13b-hf_EditAgent_results/edit_checker/3d_4.py" not in output_file:
                os.system("python3 " + output_file)

        except Exception as e:
            print(f"[SKIP] ({output_file}) Unexpected error: {traceback.format_exc()}")
            skip_stats["other_exception"] += 1
            skipped_files.append({
                "file": output_file,
                "reason": "other_exception",
                "model_response": str(raw_model_response)
            })
            continue

    skip_log_path = os.path.join(output_dir, f"skipped_rank{rank}.json")
    with open(skip_log_path, "w") as f:
        json.dump({
            "skip_stats": skip_stats,
            "skipped_files": skipped_files
        }, f, indent=2)
    
    print(f"[RANK {rank}] Skipped summary: {skip_stats}")


def collect_successful_py_files(output_dir):
    # 1. 모든 .py, .pdf 파일 이름 수집
    py_files = [f for f in os.listdir(output_dir) if f.endswith(".py")]
    pdf_files = [f for f in os.listdir(output_dir) if f.endswith(".pdf")]

    # 2. pdf가 있는 파일들의 base 이름만 추출
    pdf_base_names = set(f.replace(".pdf", "") for f in pdf_files)

    # 3. pdf가 있는 py 파일만 필터링
    successful_py_files = [f for f in py_files if f.replace(".py", "") in pdf_base_names]

    print(f"✅ PDF가 성공적으로 생성된 .py 파일 수: {len(successful_py_files)}")
    return successful_py_files




            
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input_file", type=str, default=""
    )
    parser.add_argument(
        "--template_type", type=str, default="direct"
    )
    args = parser.parse_args()
    input_file = args.input_file
    template_type = args.template_type
    print("input_file", input_file)

    data = pd.read_json(args.input_file, lines=True)

    output_dir = input_file.replace(".json", "") + "/" + template_type + "_checker"

    if os.path.exists(output_dir):
        # remove the output_dir
        os.system("rm -r " + output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("output_dir", output_dir)


    processes = []
    num_processes = 20
    for rank in range(num_processes):
        p = Process(target=_muti_process_run, args=(rank, data, num_processes))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # get all python files in the output_dir
    py_files = [f for f in os.listdir(output_dir) if f.endswith(".py")]
    print("Total Python Files", len(py_files))
    
    # get all pdf files in the output_dir
    pdf_files = [f for f in os.listdir(output_dir) if f.endswith(".pdf")]
    print("Total PDF Files", len(pdf_files))
    
    # save the count to a json file
    count = {
        "py_files": len(py_files),
        "pdf_files": len(pdf_files),
        f"{len(pdf_files)} / {len(py_files)} * 100": (len(pdf_files)/len(py_files))*100 if len(py_files) > 0 else 0,
        f"{len(pdf_files)} / 600 * 100": (len(pdf_files)/600)*100
    }
    with open(output_dir + "/count.json", "w") as f:
        json.dump(count, f)

    successful_py_files = collect_successful_py_files(output_dir)
    
    with open(os.path.join(output_dir, "successful_py_files.json"), "w") as f:
        json.dump(successful_py_files, f, indent=2)

    # all_skipped = []
    # combined_stats = {
    #     "json_parse_error": 0,
    #     "no_code_block": 0,
    #     "part2_not_found": 0,
    #     "other_exception": 0
    # }
    
    # for path in glob.glob(output_dir + "/skipped_rank*.json"):
    #     with open(path) as f:
    #         skip_data = json.load(f)
    #         all_skipped.extend(skip_data["skipped_files"])
    #         for k in combined_stats:
    #             combined_stats[k] += skip_data["skip_stats"].get(k, 0)
    converted_path = "/root/SOJUNG_STUFF/ChartMimic/results/direct/converted_result.json"
    with open(converted_path) as f:
        converted_data = json.load(f)
    converted_map = {os.path.basename(item["file"]): item["response"] for item in converted_data}

    # Collect all skipped entries and enrich with model_response from converted_results.json
    all_skipped = []
    combined_stats = {
        "json_parse_error": 0,
        "no_code_block": 0,
        "part2_not_found": 0,
        "other_exception": 0
    }

    for path in glob.glob(output_dir + "/skipped_rank*.json"):
        with open(path) as f:
            skip_data = json.load(f)
            for file_entry in skip_data["skipped_files"]:
                base_name = os.path.basename(file_entry["file"]).replace(".py", ".pdf")
                file_entry["model_response"] = converted_map.get(base_name, "")
                all_skipped.append(file_entry)
            for k in combined_stats:
                combined_stats[k] += skip_data["skip_stats"].get(k, 0)

    
    # Print result
    print("======== SKIP SUMMARY ========")
    print(json.dumps(combined_stats, indent=2))
    print(f"Total skipped files: {len(all_skipped)}")
    
    with open(os.path.join(output_dir, "skipped_summary.json"), "w") as f:
        json.dump({
            "skip_stats": combined_stats,
            "skipped_files": all_skipped
        }, f, indent=2)