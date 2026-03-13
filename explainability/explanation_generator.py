"""
explanation_generator.py

Explainable Decision Engine for the AI Resume Analyzer.
Generates human-readable reasoning for each candidate's ranking,
explaining WHY the system scored them the way it did.

Output: outputs/final_ranked_candidates_explained.csv
"""

import pandas as pd
import os
import ast
import sys

# Resolve paths relative to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

RANKED_FILE = os.path.join(PROJECT_ROOT, "outputs", "final_ranked_candidates.csv")
ENTITIES_FILE = os.path.join(PROJECT_ROOT, "datasets", "processed", "resume_entities.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "outputs", "final_ranked_candidates_explained.csv")

# Import the inference engine for category inference
from knowledge_base.skill_inference_engine import infer_skills


def generate_explanation(candidate_row, rank, resume_skills, experience):
    """
    Generate a human-readable explanation for a candidate's ranking.

    Args:
        candidate_row: Row from the ranked candidates DataFrame.
        rank: The candidate's rank position (1-based).
        resume_skills: List of skills extracted from the resume.
        experience: Experience string from the resume.

    Returns:
        str: Explanation string.
    """
    reasons = []

    ml_score = candidate_row["ML_Score"]
    skill_score = candidate_row["Skill_Score"]
    exp_score = candidate_row["Exp_Score"]
    final_score = candidate_row["Final_Score"]

    # ML similarity reasoning
    if ml_score >= 0.5:
        reasons.append(f"Strong ML text similarity ({ml_score:.2f})")
    elif ml_score >= 0.2:
        reasons.append(f"Moderate ML text similarity ({ml_score:.2f})")
    elif ml_score > 0:
        reasons.append(f"Low ML text similarity ({ml_score:.2f})")
    else:
        reasons.append("No ML text similarity")

    # Skill match reasoning with inference details
    if skill_score >= 0.8:
        reasons.append(f"Excellent skill alignment ({skill_score:.2f})")
    elif skill_score >= 0.5:
        reasons.append(f"Strong skill alignment ({skill_score:.2f})")
    elif skill_score >= 0.2:
        reasons.append(f"Partial skill match ({skill_score:.2f})")
    elif skill_score > 0:
        reasons.append(f"Weak skill match ({skill_score:.2f})")
    else:
        reasons.append("No matching skills")

    # Show inferred categories if skills exist
    if resume_skills:
        inference = infer_skills(resume_skills)
        inferred_cats = inference.get("all_inferred_categories", [])
        if inferred_cats:
            reasons.append(f"Inferred domains: {', '.join(inferred_cats[:4])}")

    # Experience reasoning
    if exp_score == 1.0:
        reasons.append(f"Experience requirement satisfied ({experience})")
    elif exp_score >= 0.5:
        reasons.append(f"Partially meets experience requirement ({experience})")
    elif exp_score > 0:
        reasons.append(f"Below experience requirement ({experience})")
    else:
        reasons.append("No experience data")

    return " | ".join(reasons)


def create_explanations():
    """Generate explanations for all ranked candidates."""

    print("Loading ranked candidates...")
    df = pd.read_csv(RANKED_FILE)

    print("Loading resume entities...")
    entities_df = pd.read_csv(ENTITIES_FILE)

    explanations = []
    skills_col = []
    inferred_col = []

    total = len(df)
    for idx, (_, row) in enumerate(df.iterrows()):
        resume_id = int(row["Resume_ID"])

        # Get skills and experience for this candidate
        if resume_id < len(entities_df):
            skills_str = entities_df.iloc[resume_id]["skills_extracted"]
            experience = str(entities_df.iloc[resume_id]["experience"])
            try:
                resume_skills = ast.literal_eval(skills_str) if isinstance(skills_str, str) else []
            except (ValueError, SyntaxError):
                resume_skills = []
        else:
            resume_skills = []
            experience = "0 years"

        # Generate explanation
        rank = idx + 1
        explanation = generate_explanation(row, rank, resume_skills, experience)
        explanations.append(explanation)

        # Store matched skills and inferred categories
        skills_col.append(", ".join(resume_skills[:8]) if resume_skills else "none")
        if resume_skills:
            inference = infer_skills(resume_skills)
            inferred_col.append(", ".join(inference.get("all_inferred_categories", [])[:5]))
        else:
            inferred_col.append("none")

        # Progress indicator
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{total} candidates...")

    df["Matched_Skills"] = skills_col
    df["Inferred_Categories"] = inferred_col
    df["Explanation"] = explanations
    df["Rank"] = range(1, len(df) + 1)

    # Reorder columns
    df = df[["Rank", "Resume_ID", "ML_Score", "Skill_Score", "Exp_Score",
             "Final_Score", "Matched_Skills", "Inferred_Categories", "Explanation"]]

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'=' * 60}")
    print("Explainable results generated!")
    print(f"{'=' * 60}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Total candidates: {len(df)}")

    print(f"\nTop 5 candidates with explanations:\n")
    for _, row in df.head(5).iterrows():
        print(f"  Rank #{int(row['Rank'])} (ID: {int(row['Resume_ID'])}) — Final Score: {row['Final_Score']}")
        print(f"    Skills: {row['Matched_Skills']}")
        print(f"    Inferred: {row['Inferred_Categories']}")
        print(f"    Reason: {row['Explanation']}")
        print()


if __name__ == "__main__":
    create_explanations()
