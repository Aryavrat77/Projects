#Installation
# !pip install keybert

from keybert import KeyBERT

# Initialize the KeyBERT model
model = KeyBERT('distilbert-base-nli-mean-tokens')

# Example text
text = """
        We are looking for a machine learning engineer with experience in Python, 
        TensorFlow, Docker, and AWS. Candidates should also have knowledge of 
        data pipelines, CI/CD (Jenkins), and advanced statistics.
       """

# Extract keywords
keywords = model.extract_keywords(text)

# Print the keywords
print("Keywords:")
for keyword in keywords:
    print(keyword)
