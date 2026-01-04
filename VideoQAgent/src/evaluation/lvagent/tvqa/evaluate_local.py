#lvagent/tvqa/evaluate_local.py
from pathlib import Path
from PIL import Image
import io, base64, math, re
from openai import OpenAI
import os
import json, argparse, time
from typing import Dict, List, Any, Tuple
import torch
from vllm import LLM, SamplingParams  # 新增 VLLM 依赖
import random

# --- 全局变量，用于存放主模型 ---
main_llm = None
main_tokenizer = None
sampling_params = None

def initialize_main_model(checkpoint_step: str, gpu_memory_utilization: float = 0.4):
    """初始化本地 VLLM 主模型"""
    global main_llm, main_tokenizer, sampling_params
    
    model_path = f"../tvqa_qwen_3b-step-{checkpoint_step}"
    print(f"  -> Loading main model from: {model_path}")

    
    main_llm = LLM(
        model=model_path,
        tokenizer=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True
    )
    
    # 配置生成参数
    sampling_params = SamplingParams(
        temperature=0.6,  # 与原 main_llm_generate 保持一致
        max_tokens=1024,  # 与原 main_llm_generate 保持一致
        skip_special_tokens=True,
    )
    print("✅ Main VLLM Model initialized.")

def main_llm_generate(conversation_history: str, model: str = "qwen2.5-7b") -> str:
    """
    使用本地 VLLM 模型生成响应（替换原 API 调用）
    
    Args:
        conversation_history: 输入的对话历史
        model: 模型名称（占位符，实际使用全局 main_llm）
    
    Returns:
        生成的响应文本
    """
    global main_llm, sampling_params
    
    if not all([main_llm, sampling_params]):
        raise RuntimeError("Main VLLM Model not initialized.")
    
    try:
        # 执行推理
        outputs = main_llm.generate([conversation_history], sampling_params)
        response = outputs[0].outputs[0].text.strip()
        return response
    except Exception as e:
        print(f"Main VLLM inference failed: {e}")
        return f"Error: Failed to generate response - {str(e)}"



def convert_image_to_base64_data_url(path: str) -> str | None:
    try:
        with Image.open(path) as img:
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            buf = io.BytesIO(); img.save(buf, format='JPEG')
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None

def process_and_query_seg(
    seg: dict,
    vid: str,
    text_client: OpenAI,
    vision_client: OpenAI,
    base_frame_dir: str = "../Tvqa/house_met_frames",
    model: str = "qwen3-vl-235b-a22b-instruct"
) -> str:
    messages_content = []
    frame_nums = list(range(1, 181, 15))
    print(f"  Selected {len(frame_nums)} frames for vision query: {frame_nums}")
    for fn in frame_nums:
        img_path = Path(base_frame_dir, vid, f"{fn:05d}.jpg")
        if img_path.is_file():
            url = convert_image_to_base64_data_url(str(img_path))
            if url:
                messages_content.append({
                    "type": "image_url",
                    "image_url": {"url": url}
                })

    messages_content.append({
        "type": "text",
        "text": (
            f"Images 1-{len(frame_nums)} are video frames, you can focus on the key objects and actions in these frames.\n And here is a description of what I want to know:\n"
            f"{seg['description']}"
        )
    })

    resp = vision_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": messages_content}]
    )
    return resp.choices[0].message.content
def extract_episode_prefix(clip_name: str) -> str:
    """从clip名称中提取剧集前缀（episode prefix，如 'met_s01e01'）"""
    if not clip_name:
        return ""
    parts = clip_name.split('_')
    if len(parts) >= 2:
        return '_'.join(parts[:2])  # 取前两部分并用 '_' 连接
    return parts[0]  # 如果不足两部分，返回第一部分
def build_subtitles_for_episode(clip_subtitles: Dict[str, str], episode_prefix: str) -> str:
    """为指定剧集构建拼接后的字幕，按clip顺序拼接"""
    # 找到所有匹配该剧集的clips
    matching_clips = {k: v for k, v in clip_subtitles.items() if k.startswith(episode_prefix)}
    
    # 按key排序（确保顺序）
    sorted_clips = sorted(matching_clips.items())
    
    # 拼接格式化的字幕
    formatted_subtitles = []
    for clip_key, subtitle_text in sorted_clips:
        formatted_subtitles.append(f"<{clip_key}>{subtitle_text}</{clip_key}>")
    
    return "\n".join(formatted_subtitles)

def get_subtitles_for_video(clip_subtitles: Dict[str, str], vid_name: str) -> str:
    """根据视频名获取对应的字幕块"""
    # 使用 extract_episode_prefix 提取剧集前缀
    episode_prefix = extract_episode_prefix(vid_name)
    return build_subtitles_for_episode(clip_subtitles, episode_prefix)
def parse_action_from_response(response: str) -> Tuple[str, str]:
    search_match = re.search(r'<search>(.*?)</search>', response, re.DOTALL)
    if search_match:
        content = search_match.group(1).strip()
        end_pos = search_match.end()
        response = response[:end_pos]
        return 'search', content
    
    grounding_match = re.search(r'<request_grounding>(.*?)</request_grounding>', response, re.DOTALL)
    if grounding_match:
        content = grounding_match.group(1).strip()
        end_pos = grounding_match.end()
        response = response[:end_pos]
        return 'request_grounding', content
    
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
        end_pos = answer_match.end()
        response = response[:end_pos]
        return 'answer', content
    
    return 'invalid', ''

def execute_action(action_type: str, content: str, 
                  vid: str, text_client: OpenAI, vision_client: OpenAI,
                  question_data: Dict, episode_sub_block: str, clip_subtitles: Dict[str, str],
                  bbox_data: Dict = None) -> Tuple[str, bool]:
    if action_type == 'answer':
        return f"\n<answer>{content}</answer>", True
    
    elif action_type == 'search':
        try:
            seg = {"description": content}
            information_parts = []
            try:
                vision_response = process_and_query_seg(seg, vid, text_client, vision_client,model="gpt-4o")
                information_parts.append(f"Visual Description:\n{vision_response.strip()}")
            except Exception as e:
                print(f"Vision LLM call failed: {e}")
                information_parts.append(f"Visual Description: Error - {str(e)}")
            
            combined_info = "\n".join(information_parts)
            return f'\n<information>{combined_info}</information>\n', False
        except Exception as e:
            print(f"Search action failed: {e}")
            return f'\n<information>Error: Failed to get information - {str(e)}</information>\n', False
    
    elif action_type == 'request_grounding':
        try:
            grounding_result = re_analyze_single_question_api(question_data, episode_sub_block, vid, attempt_round=1)
            if "error" in grounding_result:
                print(f"Grounding failed: {grounding_result['error']}")
                result_content = f"Grounding failed for query: {content}. Error: {grounding_result['error']}"
                return f'\n<grounding_info>{result_content}</grounding_info>\n', False
            else:
                predicted_clip = grounding_result.get("predicted_clip", vid)
                new_sub = get_subtitles_for_video(clip_subtitles, predicted_clip)
                result_content = f"<New_clip>{predicted_clip} + {new_sub}</New_clip>"
                return f'\n{result_content}\n', False
        except Exception as e:
            print(f"Grounding action failed: {e}")
            return f'\n<grounding_info>Error: Failed to perform grounding - {str(e)}</grounding_info>\n', False
    
    else:
        return '\nMy action is not correct. I need to search, request grounding, or answer.\n', False
def get_clip_subtitle(clip_subtitles: Dict[str, str], clip_name: str) -> str:
    """
    根据clip名称获取对应的单个clip字幕。
    
    Args:
        clip_subtitles: 包含所有clip字幕的字典，键为clip名称，值为字幕文本
        clip_name: 指定的clip名称（如 "s01e02_seg01_clip_01"）
    
    Returns:
        格式化的字幕字符串，形如 "<clip_name>subtitle_text</clip_name>"；
        如果未找到，返回空字符串
    """
    subtitle_text = clip_subtitles.get(clip_name, "")
    if subtitle_text:
        return f"<{clip_name}>{subtitle_text}</{clip_name}>"
    print(f"Warning: No subtitle found for clip {clip_name}")
    return ""



grounding_api = os.getenv("qdd_api")
vision_api = os.getenv("aliyun_api")
tencent_api = os.getenv("tencent_api")

grounding_client = OpenAI(
    api_key=grounding_api,
    base_url="https://api2.aigcbest.top/v1"
)

vision_client = grounding_client

def grounding_llm_generate(user_content: str, model: str = "grok-4-fast-reasoning") -> str:
    try:
        response = grounding_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.6,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Grounding API call failed: {e}")
        return "Error: Failed to generate response"

def postprocess_response(response: str) -> str:
    type_match = re.search(r'<type>.*?</type>', response, re.DOTALL)
    if type_match:
        return response[:type_match.end()]
    time_match = re.search(r'<time>.*?</time>', response, re.DOTALL)
    if time_match:
        return response[:time_match.end()]
    return response

def postprocess_main_response(response: str) -> str:
    """
    Postprocess the main LLM response by truncating it to the end of the first matched action tag,
    with priority: <answer> > <request_grounding> > <search>.
    
    Args:
        response: The raw response from the main LLM.
    
    Returns:
        The truncated response up to the end of the first matched tag, or the original response if no match.
    """
    # 先检查 <answer>，优先级最高
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_match:
        return response[:answer_match.end()]
    
    # 然后检查 <request_grounding>
    grounding_match = re.search(r'<request_grounding>(.*?)</request_grounding>', response, re.DOTALL)
    if grounding_match:
        return response[:grounding_match.end()]
    
    # 最后检查 <search>
    search_match = re.search(r'<search>(.*?)</search>', response, re.DOTALL)
    if search_match:
        return response[:search_match.end()]
    
    # 如果都没有匹配，返回原始响应
    return response

def extract_clip_content(text: str) -> str:
    match = re.search(r"<clip>(.*?)</clip>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def re_analyze_single_question_api(question_data: Dict, sub_block: str, vid: str, attempt_round: int) -> Dict:
    original_idx = question_data.get('original_idx', '?')
    print(f"      Question {original_idx}: API Analysis round {attempt_round}")
    
    prompt_content = f"""
Question: {question_data['q']}
Options:
a0: {question_data.get('a0', '')}
a1: {question_data.get('a1', '')}
a2: {question_data.get('a2', '')}
a3: {question_data.get('a3', '')}
a4: {question_data.get('a4', '')}

Subtitles: {sub_block}

The subtitles are formatted as <clip_label>subtitle_content</clip_label>, where each < > pair contains a clip label followed by its corresponding subtitle content.

Based on the question and subtitles, determine:
1. The specific clip label where the answer to this question occurs or is mentioned (output in <clip>label</clip> format)
{vid} may not contain the scene or context related to the question. Please determine a different specific clip label.
Please analyze the given question and provide the following information:
<clip>
clip_label (the specific clip where the question's answer can be found in the video)
</clip>
"""
    
    try:
        raw_response = grounding_llm_generate(prompt_content)
        print(f"        Raw response: {raw_response}")
        if raw_response.startswith("Error:"):
            return {"error": raw_response}
        
        processed_response = postprocess_response(raw_response)
        predicted_clip = extract_clip_content(processed_response)
        
        print(f"        Predicted clip: {predicted_clip}")
        
        return {
            "predicted_clip": predicted_clip,
            "raw_response": raw_response,
            "processed_response": processed_response
        }
    except Exception as e:
        print(f"        ❌ Analysis error: {e}")
        return {"error": str(e)}

def analyze_single_question_api(question_data: Dict, sub_block: str, attempt_round: int) -> Dict:
    original_idx = question_data.get('original_idx', '?')
    print(f"      Question {original_idx}: API Analysis round {attempt_round}")
    
    prompt_content = f"""
Question: {question_data['q']}
Options:
a0: {question_data.get('a0', '')}
a1: {question_data.get('a1', '')}
a2: {question_data.get('a2', '')}
a3: {question_data.get('a3', '')}
a4: {question_data.get('a4', '')}

Subtitles: {sub_block}

The subtitles are formatted as <clip_label>subtitle_content</clip_label>, where each < > pair contains a clip label followed by its corresponding subtitle content.

Based on the question and subtitles, determine:
1. The specific clip label where the answer to this question occurs or is mentioned (output in <clip>label</clip> format)
Please analyze the given question and provide the following information:
<clip>
clip_label (the specific clip where the question's answer can be found in the video)
</clip>
"""
    
    try:
        raw_response = grounding_llm_generate(prompt_content)
        print(f"        Raw response: {raw_response}")
        if raw_response.startswith("Error:"):
            return {"error": raw_response}
        
        processed_response = postprocess_response(raw_response)
        predicted_clip = extract_clip_content(processed_response)
        
        print(f"        Predicted clip: {predicted_clip}")
        
        return {
            "predicted_clip": predicted_clip,
            "raw_response": raw_response,
            "processed_response": processed_response
        }
    except Exception as e:
        print(f"        ❌ Analysis error: {e}")
        return {"error": str(e)}

def process_single_question(prompt: str, vid: str, question: str, 
                           question_data: Dict, episode_sub_block: str, clip_subtitles: Dict[str, str],
                           max_turn: int = 5) -> Dict[str, Any]:
    record = {
        "vid": vid, 
        "question": question, 
        "turns": [],
        "final_answer": "",
        "prompt": prompt
    }
    
    conversation_history = prompt
    final_answer = ""
    
    for turn in range(max_turn):
        print(f"\n{'='*60}")
        print(f"  Turn {turn + 1}/{max_turn}")
        print(f"{'='*60}")
        
        raw_response = main_llm_generate(conversation_history)
        response = postprocess_main_response(raw_response)
        print(f"🤖 LLM Response:\n{response}")
        
        action_type, content = parse_action_from_response(response)
        print(f"🏷️  Parsed action - Type: {action_type}, Content: {content[:100]}...{'...' if len(content) > 100 else ''}")
        
        turn_record = {
            "turn": turn + 1,
            "response": response,
            "action_type": action_type,
            "content": content,
            "is_done": False
        }
        
        print(f"🚀 Executing action: {action_type}")
        result_content, is_done = execute_action(
            action_type, content, vid, text_client=grounding_client, vision_client=vision_client,
            question_data=question_data, episode_sub_block=episode_sub_block, clip_subtitles=clip_subtitles
        )
        
        turn_record["result_content"] = result_content
        turn_record["is_done"] = is_done
        
        print(f"📋 Action result:\n{result_content}")
        
        if is_done and action_type == 'answer':
            final_answer = content
            print(f"🎉 Found answer in turn {turn + 1}: {final_answer}")
            record["turns"].append(turn_record)
            break
        
        conversation_history += result_content
        print(f"🔄 Updated conversation history with result content")
        
        if action_type == 'request_grounding' and "<New_clip>" in result_content:
            match = re.search(r"<New_clip>(.*?) \+", result_content, re.DOTALL)
            if match:
                vid = match.group(1).strip()
                print(f"Updated vid to: {vid}")
        
        record["turns"].append(turn_record)
        
        if turn == max_turn - 1:
            print(f"  Reached maximum turns ({max_turn})")
    
    record["final_answer"] = final_answer
    record["conversation_history"] = conversation_history
    
    return record

def run_enhanced_pipeline(checkpoint_step: str, max_turn: int = 5, 
                         gpu_memory_utilization: float = 0.4) -> None:
    # 初始化主模型
    initialize_main_model(checkpoint_step, gpu_memory_utilization)
    
    subs_path = "../Tvqa/tvqa_subtitles.json"
    with open(subs_path, encoding='utf-8') as f:
        clip_subtitles = json.load(f)

    # 加载 .jsonl 文件
    questions_path = "../Tvqa/house_met/tvqa_val_house_met_200_samples.jsonl"
    questions = []
    with open(questions_path, encoding='utf-8') as f:
        for line in f:
            # 跳过空行（如果有）
            if line.strip():
                questions.append(json.loads(line))



    results: List[Dict[str, Any]] = []
    total = 0
    consecutive_errors = 0
    max_consecutive_errors = 5

    for q in questions:
        try:
            q['original_idx'] = total
            total += 1
            print(f"Processing question {total}")
            
            episode_prefix = extract_episode_prefix(q["vid_name"])
            episode_sub_block = build_subtitles_for_episode(clip_subtitles, episode_prefix)
            
            grounding_result = analyze_single_question_api(q, episode_sub_block, attempt_round=1)
            predicted_clip = grounding_result.get("predicted_clip", q["vid_name"])
            print(f"Predicted clip for question {total}: {predicted_clip}")
            
            # 使用get_clip_subtitle获取单个clip的字幕（仅predicted_clip）
            sub_block = get_clip_subtitle(clip_subtitles, predicted_clip)
            
            initial_prompt = f"""You must follow these rules in every turn:
Reasoning First: Start by conducting your reasoning inside <reasoning>...</reasoning>. This is where you analyze the current information and decide your next step.
Choose One Action: After reasoning, you must choose exactly one of the following three actions:

Action A: If the current information is insufficient or you are somewhat uncertain, and you need to search for visual information on the current located <clipX>, then search for visual information. If your reasoning indicates that you lack necessary visual knowledge, you can call a visual engine. To do this, use the following format: <search>query</search>

Action B: If the current information is insufficient or you are somewhat uncertain, and you cannot obtain the final answer from the previous location and its possible visual information, then you need to call the grounding agent again for relocation, and output in the <request_grounding> format.

Action C: Provide the Final Answer
If your reasoning indicates that you have enough information to answer, provide the final answer inside <answer>...</answer>.
The answer must be concise and direct, in the format <answer>ax</answer>, where 'x' is the index of the selected option (e.g., <answer>a1</answer> for option a1).

question: {q['q']}
a0: {q['a0']}
a1: {q['a1']}
a2: {q['a2']}
a3: {q['a3']}
a4: {q['a4']}
<information>subtitles: {sub_block}</information>
"""
            
            record = process_single_question(
                prompt=initial_prompt,
                vid=predicted_clip,
                question=q['q'],
                question_data=q,
                episode_sub_block=episode_sub_block,
                clip_subtitles=clip_subtitles,
                max_turn=max_turn
            )
            
            record["predicted_clip"] = predicted_clip
            results.append(record)
            consecutive_errors = 0
            
            print(f"  Result: {len(record['turns'])} turns, Answer: {record['final_answer']}")
        
        except Exception as e:
            print(f"Error processing question {total}: {e}")
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                print(f"Too many consecutive errors ({consecutive_errors}), stopping...")
                break
    
    print(f"\nTotal processed: {total}")
    
    simplified_results = []
    for result in results:
        simplified_result = {
            "vid": result["vid"],
            "question": result["question"],
            "num_turns": len(result["turns"]),
            "final_answer": result["final_answer"],
            "predicted_clip": result.get("predicted_clip", "")
        }
        simplified_results.append(simplified_result)
    
    total_vision_calls = sum(sum(1 for t in result["turns"] if t["action_type"] == "search") for result in results)
    total_grounding_calls = sum(sum(1 for t in result["turns"] if t["action_type"] == "request_grounding") for result in results)
    
    correct_count = 0
    for result, q in zip(results, questions):
        gt_answer = f"a{q['answer_idx']}"
        pred_answer = result["final_answer"].strip().lower()
        if pred_answer == gt_answer.lower():
            correct_count += 1
    
    accuracy = correct_count / len(questions) if len(questions) > 0 else 0.0
    
    simplified_output = {
        "checkpoint_step": checkpoint_step,
        "model_path": f"../qwen2.5-7b-grpo_step-{checkpoint_step}",
        "gpu_memory_utilization": gpu_memory_utilization,
        "total": total,
        "max_turn": max_turn,
        "metadata": {
            "avg_turns": sum(r["num_turns"] for r in simplified_results) / len(simplified_results) if simplified_results else 0,
            "vision_calls_total": total_vision_calls,
            "grounding_calls_total": total_grounding_calls,
            "completed_questions": len([r for r in simplified_results if r["final_answer"]]),
            "completion_rate": len([r for r in simplified_results if r["final_answer"]]) / len(simplified_results) if simplified_results else 0,
            "accuracy": accuracy
        },
        "results": simplified_results
    }
    
    simplified_filename = f"./eval_new_action-{checkpoint_step}_demo_summary.json"
    with open(simplified_filename, "w", encoding="utf-8") as f:
        json.dump(simplified_output, f, ensure_ascii=False, indent=2)
    
    print(f"Summary results saved to {simplified_filename}")
    
    if simplified_results:
        metadata = simplified_output["metadata"]
        print(f"\n📊 Statistics for 7B checkpoint-{checkpoint_step}:")
        print(f"Model path: ../qwen2.5-7b-grpo_step-{checkpoint_step}")
        print(f"GPU memory utilization: {gpu_memory_utilization}")
        print(f"Average turns per question: {metadata['avg_turns']:.2f}")
        print(f"Total vision calls: {metadata['vision_calls_total']}")
        print(f"Total grounding calls: {metadata['grounding_calls_total']}")
        print(f"Vision calls per question: {metadata['vision_calls_total']/len(simplified_results):.2f}")
        print(f"Completed questions: {metadata['completed_questions']}/{len(simplified_results)}")
        print(f"Completion rate: {metadata['completion_rate']:.2%}")
        print(f"Accuracy: {metadata['accuracy']:.2%}")
        
        turn_counts = {}
        for r in simplified_results:
            turns = r["num_turns"]
            turn_counts[turns] = turn_counts.get(turns, 0) + 1
        
        print(f"\n🔄 Turn distribution:")
        for turns in sorted(turn_counts.keys()):
            count = turn_counts[turns]
            print(f"  {turns} turns: {count} questions ({count/len(simplified_results)*100:.1f}%)")
    
    # 释放 VLLM 资源
    if main_llm:
        del main_llm
        torch.cuda.empty_cache()
        print("🔄 Main VLLM resources released")

def main():
    parser = argparse.ArgumentParser(description="运行增强版流水线 - 本地VLLM主模型")
    parser.add_argument("--checkpoint_step", "-c", type=str, default="api", help="主模型的checkpoint步数")
    parser.add_argument("--max_turn", "-t", type=int, default=5, help="最大对话轮数")
    parser.add_argument("--gpu_memory_utilization", "-g", type=float, default=0.4, help="GPU内存利用率 (0.0-1.0)")
    
    args = parser.parse_args()
    
    run_enhanced_pipeline(
        checkpoint_step=args.checkpoint_step,
        max_turn=args.max_turn,
        gpu_memory_utilization=args.gpu_memory_utilization
    )

if __name__ == "__main__":
    main()