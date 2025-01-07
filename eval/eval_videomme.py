import fire
import json
from pathlib import Path
from eval.utils_videomme import VideoMMEDataset
from tools.llava_chat import LLaVA
from tools.kangaroo_chat import Kangaroo
from tools.videollava_chat import VideoLLaVA
from tools.cross_attn_chat import LLMCrossAttn
from tools.longva_chat import LongVA
from tools.qwen2_vl_chat import QWen2_VL
from tools.idefics2_chat import Idefics2
from tools.phi3_v_chat import Phi3_V
# from tools.llava_ov_chat import LLaVA_OneVision
from tqdm import tqdm

def main(
    model_type="longva",
    model_name_or_path="/cpfs/data/user/weiming/checkpoints/lmms-lab/LongVA-7B",
    data_dir="/cpfs/data/user/weiming/datasets/videomme",
    num_frames=8,
    img_shortest_edge=256,
    img_longest_edge=480,
    max_img_seq_len=16500,
    do_resize=False,
    use_subtitle=False,
    results_dir="./output/eval/videomme",
    overwrite=False,
    # generation config
    max_new_tokens=512,
    do_sample=False,
    top_k=None,
    top_p=0.9,
    temperature=0.6,
):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if model_type == "llava":
        model = LLaVA(model_name_or_path)
    elif model_type == "kangaroo":
        model = Kangaroo(model_name_or_path)
    elif model_type == "cross_attn":
        model = LLMCrossAttn(model_name_or_path)
    elif model_type == "videollava":
        model = VideoLLaVA(model_name_or_path)
    elif model_type == "longva":
        model = LongVA(model_name_or_path)
    elif model_type == "qwen2_vl":
        model = QWen2_VL(model_name_or_path)
    elif model_type == "idefics2":
        model = Idefics2(model_name_or_path)
    elif model_type == "phi3_v":
        model = Phi3_V(model_name_or_path)
    # elif model_type == "llava_ov":
    #     model = LLaVA_OneVision(model_name_or_path)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    sample_config = {
        'num_frames': num_frames,
        'sample_type': 'uniform',
        'model_patch_size': model.patch_size,
        'img_shortest_edge': img_shortest_edge,
        'img_longest_edge': img_longest_edge,
        'max_img_seq_len': max_img_seq_len,
        'do_resize': do_resize,
    }

    dataset = VideoMMEDataset(
        data_dir,
        sample_config=sample_config,
        use_subtitle=use_subtitle
    )

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
    }

    core_data = []

    model_save_path = "/".join(model_name_or_path.split("/")[-2:])
    results_file = Path(results_dir) / f"{num_frames}frames" / f"{model_save_path}.jsonl"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    if results_file.exists() and not overwrite:
        with open(results_file, "r") as rf:
            for line in rf:
                core_data.append(json.loads(line))
    else:
        with open(results_file, "w") as wf:
            pass
    
    for i in tqdm(range(len(core_data), len(dataset))):
        data = dataset[i]
        meta = data["meta"]
        di = {
            "question": data["text"][0],
            "answer": meta["questions"][0]["answer"],
            "task_type": meta["questions"][0]["task_type"],
            "video_id": meta["video_id"],
            "duration": meta["duration"],
            "domain": meta["domain"],
            "sub_category": meta["sub_category"],
            "videoID": meta["videoID"],
            "question_id": meta["questions"][0]["question_id"]
        }
        
        if model_type == "kangaroo":
            images = (data["video"], data["durations"])
        else:
            images = data["video"]
        question = data["text"][0]
        messages = [
            {
                "type": "pil_video",
                "content": images
            },
            {
                "type": "text",
                "content": f"<video> {question}",
            }
        ]
        response = model(messages, generation_config)
        response = response.lower()
        
        di["outputs"] = response
        if "the answer is" in response:
            response = response.split("the answer is")[-1].strip()
        elif "answer:" in response:
            response = response.split("answer:")[-1].strip()
        elif "the option is" in response:
            response = response.split("the option is ")[-1].strip()
        for char in response:
            if char.isalpha():
                response = char
                break
        di["correct"] = response[0] == di["answer"] or response[0] == di["answer"].lower() if len(response) > 0 else False
        
        with open(results_file, "a") as wf:
            json.dump(di, wf)
            wf.write("\n")
        core_data.append(di)
        
    # print accuracy
    task_type_dict = {}
    for item in core_data:
        task_type = item["task_type"]
        if task_type not in task_type_dict:
            task_type_dict[task_type] = {"correct": 0, "total": 0}
        task_type_dict[task_type]["total"] += 1
        if item["correct"]:
            task_type_dict[task_type]["correct"] += 1
    for task_type in task_type_dict:
        print(f"Task Type: {task_type}")
        print(f"Accuracy: {task_type_dict[task_type]['correct']} / {task_type_dict[task_type]['total']:.4f} = {task_type_dict[task_type]['correct'] / task_type_dict[task_type]['total']:.4f}")
        print()
    duration_dict = {}
    for item in core_data:
        duration = item["duration"]
        if duration not in duration_dict:
            duration_dict[duration] = {"correct": 0, "total": 0}
        duration_dict[duration]["total"] += 1
        if item["correct"]:
            duration_dict[duration]["correct"] += 1
    for duration in duration_dict:
        print(f"Duration: {duration}")
        print(f"Accuracy: {duration_dict[duration]['correct']} / {duration_dict[duration]['total']:.4f} = {duration_dict[duration]['correct'] / duration_dict[duration]['total']:.4f}")
        print()
    all_correct = sum([task_type_dict[task_type]["correct"] for task_type in task_type_dict])
    all_total = sum([task_type_dict[task_type]["total"] for task_type in task_type_dict])
    print(f"Overall Accuracy: {all_correct} / {all_total:.4f} = {all_correct / all_total:.4f}")
        
        
if __name__ == "__main__":
    fire.Fire(main)