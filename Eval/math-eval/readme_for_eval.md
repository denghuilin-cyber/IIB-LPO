Step1:

sh /nas/dhl/Eval/math-eval/my_eval.sh

先运行这个， rollout问题 并 提取答案， 保存为jsonl文件。

再运行：

Step2:

python /nas/dhl/Eval/math-eval/calculate_scores.py

这个脚本里面的文件，需要设置成 第一步生成的那个文件

计算指标，需要设置 

  n = 1

  Ks = [1]  一致就行，如果只想算 pass@k 如果算别的，需要改一下。