import os
import json
import argparse
import re
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from mathruler.grader import extract_boxed_content, grade_answer
os.environ["NCCL_DEBUG"] = "WARN"


def extract_final_answer(text):
    if "####" in text:
        parts = text.split("####")
        answer = parts[-1].strip()
        # 移除可能的结束标签如 <|im_end|> 或其他 <...>
        answer = re.sub(r'<.*?>', '', answer).strip()
        return answer
    return None


def prepare_data(example, prompt_key,chat_template = "qwen3"):
    # qwen_boxed_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n"
    if chat_template == "qwen2.5":
        qwen_boxed_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input}\n Let's think step by step and output the final answer after \"####\". You must stop after final answer. <|im_end|>\n<|im_start|>assistant\n"
    elif chat_template == "qwen3":
        qwen_boxed_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input}\nr here]  Let's think step by step and output the final answer after \"####\". You must stop after final answer. <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
    example['prompt'] = qwen_boxed_prompt.replace("{input}", example[prompt_key])

        #question_with_instruction = question + " Let's think step by step and output the final answer after \"####\"."
        
        #question_with_instruction = question + "Let's think step by step and provide your response in the following format: [Your step-by-step thinking here] #### [Your final answer here]. You must stop after final answer."
        
    return example
def get_resume_index(output_file_path):
    """
    Checks an output file and returns the index from which to resume.
    """
    if not os.path.exists(output_file_path):
        return 0
   
    last_question_id = -1
    with open(output_file_path, 'r') as f:
        lines = f.readlines()
        if not lines:
            return 0
       
        # Fast check: read the last line
        try:
            last_line = lines[-1]
            last_entry = json.loads(last_line)
            # question_id is the 0-based index in the original dataset
            last_question_id = last_entry.get('question_id', -1)
        except (json.JSONDecodeError, IndexError):
            # If file is corrupted or empty, re-scan from beginning to be safe
            print("Warning: Could not parse the last line. Re-scanning the whole file to find the last valid entry.")
            for line in reversed(lines):
                try:
                    entry = json.loads(line)
                    last_question_id = entry.get('question_id', -1)
                    if last_question_id != -1:
                        break # Found a valid entry
                except json.JSONDecodeError:
                    continue # Skip corrupted lines
   
    # The next index to process is last_question_id + 1
    return last_question_id + 1
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--datasets", type=str)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--do_sample", type=bool, default=True) # Note: this argument isn't used by vLLM SamplingParams in this script
    parser.add_argument("--num_generation", type=int, default=1)
    parser.add_argument("--dataset_num_proc", type=int, default=1)
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--chat_template", type=str, default="qwen2.5")
    # The resume_id argument is no longer needed for automatic resumption
    # parser.add_argument("--resume_id", type=int, default=0)
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.model_name):
        print(f"Model {args.model_name} not found. Skip.")
        return
    # Load the model and tokenizer
    print(f"Loading model {args.model_name}")
    llm = LLM(args.model_name, tensor_parallel_size=args.num_gpus, dtype="bfloat16", gpu_memory_utilization=0.9, trust_remote_code=True)
    sampling_params = SamplingParams(
        n=args.num_generation,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens
    )
    # Load the dataset
    datasets = args.datasets.split(",")
    for dataset_name in datasets:
        # Define output file path early to check for existence
        output_file_name = dataset_name.split("/")[-1] + '-' + args.split + '-temp_' + str(args.temperature) + "-top_p_" + str(args.top_p) + "-top_k_" + str(args.top_k) + f'{args.comment}.jsonl'
        output_file_path = os.path.join(args.output_dir, output_file_name)
        os.makedirs(args.output_dir, exist_ok=True)
        full_dataset = load_dataset(dataset_name, split=args.split)
       
        # --- Automatic Resumption Logic ---
        resume_index = get_resume_index(output_file_path)
       
        if resume_index >= len(full_dataset):
            print(f"Dataset '{dataset_name}' is already fully processed. Skipping.")
            continue
        if resume_index > 0:
            print(f"Resuming from index {resume_index} for dataset '{dataset_name}'.")
       
        # Select the remaining part of the dataset to process
        dataset_to_process = full_dataset.select(range(resume_index, len(full_dataset)))
       
        # dataset_to_process = dataset_to_process.filter(lambda example: example['level'] == 'Level 5') # Optional filter
       
        if "math" in dataset_name.lower():
            prompt_key = "problem"
            answer_key = "solution"
        elif "aime" in dataset_name.lower() or "amc23" in dataset_name.lower():
            prompt_key = "problem"
            answer_key = "answer"
        else: # Add a default case
            prompt_key = "problem"
            answer_key = "answer"
        dataset_to_process = dataset_to_process.map(lambda x: prepare_data(x, prompt_key, args.chat_template), num_proc=args.dataset_num_proc)
        # Use 'a' for append mode if resuming, 'w' for write mode if starting new
        open_mode = 'a' if resume_index > 0 else 'w'
        with open(output_file_path, open_mode) as f:
            for i in tqdm(range(0, len(dataset_to_process), args.batch_size), desc=f"Processing {dataset_name}"):
                batch = dataset_to_process[i:i + args.batch_size]
                inputs = batch["prompt"]
                answers = batch[answer_key]
                # Generate the answer
                outputs = llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False) # use_tqdm=False to avoid nested progress bars
                results = [[_.outputs[l].text for l in range(len(_.outputs))] for _ in outputs]
                assert len(results[0]) == args.num_generation, f"Number of generations is not equal to {args.num_generation}, got {len(results[0])}"
                # Process the results
                for j, (inp, q, a, r) in enumerate(zip(inputs, batch[prompt_key], answers, results)):
                    for k in range(args.num_generation):
                        # Calculate the correct global question_id
                        current_question_id = resume_index + i + j
                       
                        qa_pair = {
                            "prompt": inp,
                            "vanilla_response": r[k],
                            "question": q,
                            "answer": a, # 是数据集的标准答案
                            "question_id": current_question_id,
                            "generation_id": k,
                        }
                        qa_pair["response"] = r[k]
                        if "math" in dataset_name.lower():
                            gold_answer = extract_boxed_content(a)
                            pred_answer = extract_final_answer(qa_pair["response"])
                        elif "amc23" in dataset_name.lower() or "aime" in dataset_name.lower():
                            gold_answer = a
                            pred_answer = extract_final_answer(qa_pair["response"])
                        else: # Default grading
                             gold_answer = a
                             pred_answer = extract_final_answer(qa_pair["response"])
                        qa_pair["label"] = grade_answer(pred_answer, gold_answer)
                        qa_pair["gold_answer"] = gold_answer
                        qa_pair["pred_answer"] = pred_answer
                        f.write(json.dumps(qa_pair) + '\n')
                f.flush()
if __name__ == "__main__":
    main()