import argparse
import json
import numpy as np
from collections import defaultdict, Counter # <-- NEW: Import Counter
from tqdm import tqdm
from mathruler.grader import extract_boxed_content, grade_answer

def calculate_metrics(file_path, k, n):
    """
    Calculates various metrics: Unbiased Pass@k, standard Pass@k, Average@k, 
    MajorityCorrect@n, and Maj@n(Vote).

    Args:
        file_path (str): Path to the JSONLines file.
        k (int): The 'k' for pass@k and average@k calculations.
        n (int): The total number of samples generated per question.
    """
    # 1. Read the JSONLines file
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    # 2. Group predictions by question_id
    grouped_data = defaultdict(list)
    average_data = defaultdict(list)
    
    for example in data:
        question_id = example["question_id"]
        if len(grouped_data[question_id]) < n:
            grouped_data[question_id].append(example)
        if len(average_data[question_id]) < k:
            average_data[question_id].append(example)

    # 3. Initialize counters
    total_pass_k_prob = 0
    total_pass_k_prob_2 = 0
    total_avg_k_prob = 0
    # ### MODIFIED ###: Renamed the old Maj@n counter
    maj_correct_questions = 0
    # ### NEW ###: Counter for the new, correct Maj@n (Vote)
    maj_vote_correct_questions = 0  
    total_questions = 0
    
    # 4. Calculate metrics that require all `n` samples
    for question_id, examples in grouped_data.items():
        if len(examples) != n:
            print(f"Warning: Question {question_id} has {len(examples)} samples, but expected {n}. Skipping for unbiased pass@k and maj@n metrics.")
            continue
        
        # This is the count of correct generations among all n samples.
        c = sum(1 for ex in examples if ex["label"])
        
        # --- ### MODIFIED ###: MajorityCorrect@n Calculation ---
        # This checks if more than half of the generations are correct.
        if c >= n / 2:
            maj_correct_questions += 1
        
        # --- ### NEW ###: Maj@n (Majority Vote) Calculation ---
        # This finds the most frequent answer and checks if IT is correct.
        # 1. Get all predicted answers for the current question.
        #    Assumes the field is named "pred_answer" as per your format.
        predicted_answers = [ex.get("pred_answer") for ex in examples]
        
        # 2. Count the frequency of each answer.
        if not predicted_answers: # Handle empty case
            continue
        answer_counts = Counter(predicted_answers)
        
        # 3. Find the most common answer. most_common(1) returns a list like [('answer', count)]
        most_common_answer, _ = answer_counts.most_common(1)[0]
        
        # 4. Check if this most common answer is correct.
        #    We find one instance of this answer and check its "label".
        #    The "label" should be consistent  for the same predicted answer within a question.
        is_majority_answer_correct = False
        for ex in examples:
            if ex.get("pred_answer") == most_common_answer:
                if ex["label"]: # If the label for this answer is True
                    is_majority_answer_correct = True
                break # Found the answer, no need to search further
        
        if is_majority_answer_correct:
            maj_vote_correct_questions += 1
        # --- End of new logic ---

        # --- Unbiased Pass@k Calculation ---
        if n - c < k:
            prob = 1.0
        else:
            prob = 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))        
        total_pass_k_prob += prob
        total_questions += 1

    # 5. Calculate metrics that only require `k` samples (Standard Pass@k, Average@k)
    processed_questions_for_avg = 0
    for question_id, examples in average_data.items():
        if question_id not in grouped_data or len(grouped_data[question_id]) != n:
            continue
            
        num_samples_for_avg = min(len(examples), k)
        if num_samples_for_avg == 0:
            continue

        c = sum(1 for ex in examples if ex["label"])
        
        if c > 0:
            total_pass_k_prob_2 += 1
        
        total_avg_k_prob += c / num_samples_for_avg
        processed_questions_for_avg += 1

    # 6. Calculate and print the final results
    # ### MODIFIED ###: Calculate and display both majority metrics
    if total_questions > 0:
        avg_pass_k = total_pass_k_prob / total_questions
        maj_correct_accuracy = maj_correct_questions / total_questions
        maj_vote_accuracy = maj_vote_correct_questions / total_questions
    else:
        avg_pass_k = 0
        maj_correct_accuracy = 0
        maj_vote_accuracy = 0

    if processed_questions_for_avg > 0:
        avg_avg_k = total_avg_k_prob / processed_questions_for_avg
        avg_pass_k_2 = total_pass_k_prob_2 / processed_questions_for_avg
    else:
        avg_avg_k = 0
        avg_pass_k_2 = 0

    print(f"Questions Number: {total_questions}")
    print(f"Unbiased Pass@{k}/{n}: {avg_pass_k:.3f}")
    print(f"Average@{k}: {avg_avg_k:.3f}")
    print(f"Standard Pass@{k}: {avg_pass_k_2:.3f}")
    print(f"MajorityCorrect@{n} (c >= n/2): {maj_correct_accuracy:.3f}")
    print(f"Maj@{n} (Vote): {maj_vote_accuracy:.3f}")

    return avg_pass_k, maj_correct_accuracy, maj_vote_accuracy


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--file_path", type=str, required=True, help="Path to the JSONLines file")
    # parser.add_argument("--n", type=int, required=True, help="Total number of generations per problem")
    # parser.add_argument("--k", nargs='+', type=int, required=True, help="List of k values for pass@k")
    # args = parser.parse_args()
    
    file_path = "/nas/dhl/Eval/math-eval/“/nas/dhl/Eval/Eval_Output/Qwen3-4B-Instruct-8192”/MATH-500-test-temp_0.0-top_p_0.95-top_k_-1.jsonl"
    n = 1
    Ks = [1] 
   
    for k_val in tqdm(Ks, desc="Calculating metrics for different K"):
        print("-" * 80)
        print(f"Calculating for k={k_val} with n={n}")
        calculate_metrics(file_path=file_path, k=k_val, n=n)