Resume Parser and Skills Extraction

python3.10 -m venv .venv

pip install pdfminer.six spacy
python -m spacy download en_core_web_sm

file.xlsx contains the extracted skills for ground truth and model output

pip install pandas openpyxl

Precision: Of all skills the model extracted, what percentage were correct?
Formula: correctly_extracted / (correctly_extracted + extra)

Recall: Of all skills that should have been extracted, what percentage were actually extracted?
Formula: correctly_extracted / (correctly_extracted + missing)

F1 Score: Harmonic mean of precision and recall
Formula: 2 * (precision * recall) / (precision + recall)

Accuracy: What percentage of ground truth skills were correctly extracted?
Formula: correctly_extracted / total_ground_truth
