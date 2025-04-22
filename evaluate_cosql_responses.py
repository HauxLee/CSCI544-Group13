import json
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import language_tool_python
from collections import Counter



json_files = [
    "cosql_test_results.json",
    "logs/run_cosql_test_0412_153437.json",
    "logs/run_cosql_test_0412_154741.json",
    "logs/run_cosql_test_0414_182731.json"
]

# ==========  ==========

def tokenize(text):
    """"""
    return re.findall(r"\w+|\S", text.lower())

def bounded_lcr(pred, ref):
    pred_counts = Counter(pred)
    ref_counts = Counter(ref)
    matched = sum(min(pred_counts[w], ref_counts[w]) for w in ref_counts)
    return (matched / len(ref)) * 100 if ref else 0

# ==========  ==========
tool = language_tool_python.LanguageTool("en-US")
smooth_fn = SmoothingFunction().method1

bleu_scores = []
lcr_scores = []
grammar_errors = []
total_samples = 0

valid_bleu_scores = []
# ==========  ==========
for file in json_files:
    with open(file, "r") as f:
        entries = json.load(f)

    for entry in entries:
        pred_nl = entry.get("agent_nl_response")
        ref_nl = entry.get("ground_truth_nl_response")
        agent_sql_result = entry.get("agent_sql_execution_result")
        agent_generated_sql = entry.get("agent_generated_sql")
        gt_result = entry.get("ground_truth_sql_result")

        if pred_nl and ref_nl and agent_sql_result and gt_result and agent_generated_sql:
            total_samples += 1

           
            ref_tokens = tokenize(ref_nl)
            pred_tokens = tokenize(pred_nl)
            sql_pred_tokens = tokenize(agent_generated_sql)
            bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_fn)
            valid_bleu_scores.append(bleu)


            # LCR
            lcr = bounded_lcr(pred_nl, agent_generated_sql)
            #lcr = (len(set(pred_tokens) & set(sql_pred_tokens)) / len(sql_pred_tokens)) * 100
            lcr_scores.append(lcr)

            # Grammar
            grammar_matches = tool.check(pred_nl)
            grammar_errors.append(len(grammar_matches))

# ==========  ==========
print("\n========== CoSQL NL Response Evaluation ==========")
print(f"Total Evaluated Responses: {total_samples}")
print(f"Average BLEU Score: {round(sum(valid_bleu_scores) / len(valid_bleu_scores) * 100, 2)}")
print(f"Average LCR (%): {round(sum(lcr_scores) / total_samples, 2)}")
print(f"Average Grammar Errors per Response: {round(sum(grammar_errors) / total_samples, 2)}")