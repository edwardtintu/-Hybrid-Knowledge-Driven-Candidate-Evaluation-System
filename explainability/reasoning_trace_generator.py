import json

def generate_reasoning_trace(
    candidate,
    matched_skills,
    missing_skills,
    rules_fired,
    ml_score,
    rule_score,
    graph_score,
    hybrid_score
):
    """
    Generates an explicit, explainable Reasoning Trace for 
    how the AI formulated its final Hybrid Analysis Score.
    This fulfills the 'Reasoning Trace' patent requirement.
    """
    
    trace = f"Reasoning Trace for Candidate: {candidate}\n"
    trace += "=" * 50 + "\n\n"
    
    trace += "1. Skill Extraction Analysis\n"
    trace += f"   ✅ Matched Skills: {', '.join(matched_skills) if matched_skills else 'None'}\n"
    trace += f"   ❌ Missing Skills: {', '.join(missing_skills) if missing_skills else 'None'}\n\n"
    
    trace += "2. Formal Rule Engine Execution\n"
    if rules_fired and isinstance(rules_fired, list):
        trace += f"   Fired Rules: {', '.join(rules_fired)}\n\n"
    else:
        trace += "   Fired Rules: None\n\n"
        
    trace += "3. Final Hybrid Score Calculation\n"
    trace += f"   ML Similarity Score:   {ml_score:.4f} (Weight: 0.5)\n"
    trace += f"   Graph Inference Score: {graph_score:.4f} (Weight: 0.2)\n"
    trace += f"   Rule Logic Score:      {rule_score:.4f} (Weight: 0.3)\n"
    trace += "   ----------------------------------------\n"
    trace += f"   Ultimate Score:        {hybrid_score:.4f}\n"

    return trace
