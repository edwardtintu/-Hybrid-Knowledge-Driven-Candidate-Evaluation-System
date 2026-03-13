"""
decision_explanation_engine.py

Generates a human-readable reasoning trace for final candidate rankings.
Combines the ML similarity score, Rule Execution score, and specific
Rules Fired into a clear explanation string.

Input:  outputs/hybrid_final_ranking.csv
Output: outputs/hybrid_ranking_explained.csv
"""

import pandas as pd
import os
import sys

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

INPUT_FILE = os.path.join(PROJECT_ROOT, "outputs", "hybrid_final_ranking.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "outputs", "hybrid_ranking_explained.csv")


def generate_explanation(row):
    """
    Generate a decision explanation for a single candidate.
    """
    explanation = []
    
    # Extract scores
    ml_score = float(row.get("ML_Score", 0))
    rule_score = float(row.get("Rule_Score", 0))
    hybrid_score = float(row.get("Hybrid_Score", 0))
    rules_fired = str(row.get("Rules_Fired", ""))
    rule_details = str(row.get("Rule_Details", ""))
    
    # 1. Base ML performance
    explanation.append(f"ML similarity score = {ml_score:.2f}")
    
    # 2. Base Rule performance
    explanation.append(f"Rule reasoning score = {rule_score:.2f}")
    
    # 3. Rules triggered
    if rules_fired and rules_fired.lower() != "none" and str(rules_fired) != "nan":
        explanation.append(f"Rules triggered: {rules_fired}")
        
    # 4. Eligibility summary based on rule score
    if rule_score > 0.8:
        explanation.append("Exceptional rule-based candidate eligibility")
    elif rule_score > 0.5:
        explanation.append("Strong rule-based candidate eligibility")
    elif rule_score > 0.2:
        explanation.append("Moderate rule-based candidate eligibility")
    else:
        explanation.append("Weak rule-based candidate eligibility")
        
    # 5. Extract specific details if R9 (Top Flag) is present
    if "R9" in rules_fired:
        explanation.append("Flagged as Top Candidate by Rule Engine")
        
    return " | ".join(explanation)


def create_decision_explanations():
    print(f"Loading hybrid ranking data from: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Run hybrid_decision_fusion.py first.")
        return

    explanations = []
    
    total = len(df)
    for idx, row in df.iterrows():
        exp = generate_explanation(row)
        explanations.append(exp)
        
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{total} explanations...")

    df["Decision_Explanation"] = explanations
    
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n{'=' * 60}")
    print("Decision explanations generated!")
    print(f"{'=' * 60}")
    print(f"Output saved to: {OUTPUT_FILE}")
    print(f"Total candidates explained: {len(df)}")
    
    print("\nTop 3 Candidate Explanations:\n")
    for _, r in df.head(3).iterrows():
        print(f"Rank #{int(r['Rank'])} (ID: {int(r['Resume_ID'])}) — Hybrid Score: {r['Hybrid_Score']:.4f}")
        for part in r['Decision_Explanation'].split(' | '):
            print(f"  - {part}")
        print()


if __name__ == "__main__":
    create_decision_explanations()
