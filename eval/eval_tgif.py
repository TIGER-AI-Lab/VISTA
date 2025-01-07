import fire
import json
from pathlib import Path
from eval.utils_tgif import TGIFDataset
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
    data_dir="/cpfs/data/user/weiming/datasets/TGIF_Zero_Shot_QA/mp4",
    json_path="/cpfs/data/user/weiming/datasets/TGIF_Zero_Shot_QA/tgif_frameqa.json",
    num_frames=64,
    img_shortest_edge=256,
    img_longest_edge=480,
    max_img_seq_len=40000,
    do_resize=False,
    results_dir="./output/eval/tgif",
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
        'do_resize': do_resize,
        'max_img_seq_len': max_img_seq_len,
    }
    print(sample_config)

    dataset = TGIFDataset(
        data_dir,
        json_path,
        sample_config=sample_config,
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
            "question": data["text"],
            "question_raw": meta["question"],
            "answer": meta["answer"],
            "id": meta["question_id"],
            "video_id": meta["video_name"],
        }
        
        if model_type == "kangaroo":
            images = (data["video"], data["durations"])
        else:
            images = data["video"]
        question = data["text"]
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
        
        di["outputs"] = response
        
        with open(results_file, "a") as wf:
            json.dump(di, wf)
            wf.write("\n")
        core_data.append(di)        
        
if __name__ == "__main__":
    fire.Fire(main)