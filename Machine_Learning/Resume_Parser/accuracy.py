import pandas as pd

# Load Excel file (Replace 'your_file.xlsx' with actual file name)
file_path = "file.xlsx"
df = pd.read_excel(file_path)

# Ensure column names match your dataset
ground_truth_col = df.columns[1]  # First column (Correct Skills)
model_output_col = df.columns[2]  # Second column (Extracted by Model)

# Function to clean and normalize skills
def preprocess_skills(skill_text):
    if pd.isna(skill_text):  # Handle empty cells
        return set()
    return set(skill.strip().lower().replace("-", "").strip() for skill in skill_text.split("\n"))

# Compare skills row by row and calculate metrics
results = []
total_correct = total_missing = total_extra = total_ground_truth = 0

for index, row in df.iterrows():
    ground_truth_skills = preprocess_skills(row[ground_truth_col])
    model_extracted_skills = preprocess_skills(row[model_output_col])
    
    correct = ground_truth_skills.intersection(model_extracted_skills)  # ✅ Correctly extracted
    missing = ground_truth_skills.difference(model_extracted_skills)  # ❌ Missing
    extra = model_extracted_skills.difference(ground_truth_skills)  # ⚠️ Extra

    # Count skills for numerical evaluation
    total_correct += len(correct)
    total_missing += len(missing)
    total_extra += len(extra)
    total_ground_truth += len(ground_truth_skills)

    # Compute Precision, Recall, F1 Score for this row
    precision = len(correct) / (len(correct) + len(extra)) if (len(correct) + len(extra)) > 0 else 0
    recall = len(correct) / (len(correct) + len(missing)) if (len(correct) + len(missing)) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = len(correct) / len(ground_truth_skills) if len(ground_truth_skills) > 0 else 0

    results.append({
        "Ground Truth": row[ground_truth_col],
        "Model Output": row[model_output_col],
        "Correctly Extracted": ", ".join(correct),
        "Missing Skills": ", ".join(missing),
        "Extra Skills": ", ".join(extra),
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1 Score": round(f1_score, 2),
        "Accuracy": round(accuracy, 2)
    })

# Compute overall metrics
overall_precision = total_correct / (total_correct + total_extra) if (total_correct + total_extra) > 0 else 0
overall_recall = total_correct / (total_correct + total_missing) if (total_correct + total_missing) > 0 else 0
overall_f1_score = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
overall_accuracy = total_correct / total_ground_truth if total_ground_truth > 0 else 0

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Add overall metrics to the results DataFrame
overall_metrics = pd.DataFrame([{
    "Ground Truth": "OVERALL METRICS",
    "Model Output": "",
    "Correctly Extracted": total_correct,
    "Missing Skills": total_missing,
    "Extra Skills": total_extra,
    "Precision": round(overall_precision, 2),
    "Recall": round(overall_recall, 2),
    "F1 Score": round(overall_f1_score, 2),
    "Accuracy": round(overall_accuracy, 2)
}])

results_df = pd.concat([results_df, overall_metrics], ignore_index=True)

# Save comparison results to a new Excel file
output_file = "skills_comparison_results.xlsx"
results_df.to_excel(output_file, index=False)

print(f"Comparison completed! Results saved to {output_file}.")
print(f"Overall Precision: {round(overall_precision, 2)}")
print(f"Overall Recall: {round(overall_recall, 2)}")
print(f"Overall F1 Score: {round(overall_f1_score, 2)}")
print(f"Overall Accuracy: {round(overall_accuracy, 2)}")
