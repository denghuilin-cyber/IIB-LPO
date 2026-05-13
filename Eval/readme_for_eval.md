
**Step 1:**
```bash
sh /nas/dhl/Eval/math-eval/my_eval.sh
```
Run this script first to perform the rollout, extract the answers, and save the results as a `.jsonl` file.

**Step 2:**
Then run:
```bash
python /nas/dhl/Eval/math-eval/calculate_scores.py
```
Inside this script, you need to set the input file path to the `.jsonl` file generated in Step 1.

To calculate the metrics, set the following parameters:
`n = 1`
`Ks = [1]`
Keep these values consistent. This setup is sufficient if you only want to calculate `pass@k`. If you need to calculate other metrics, you will need to adjust these parameters accordingly.



> *Note on Datasets:* The evaluation datasets will be automatically downloaded during the run. They are saved in the `data/` directory, following a structure like this:
> ```text
> Eval/math-eval/data/
> ├── acm23/
> │   └── test.jsonl
> ├── aime2024/
> │   └── test.jsonl
> ├── aime2025/
> │   └── test.jsonl
> └── math500/
>     └── test.jsonl
> ```
