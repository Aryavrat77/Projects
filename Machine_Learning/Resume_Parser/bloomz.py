import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "bigscience/bloomz-560m"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model in float16 precision if you have a GPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # use float32 if you're on CPU only
    device_map="cpu"          # automatically map model layers to your GPU(s)
)


system_prompt = (
    "You are an assistant specialized in extracting technical skills from job descriptions. "
    "When given a job description, your goal is to list only the technical skills, tools, and "
    "technologies mentioned, in bullet form, without extra commentary."
)

few_shot_examples = """
Example 1:
Job Description: "We are hiring a software developer proficient in Java, Spring Boot, SQL, and Docker."
Extracted Keywords:
- Java
- Spring Boot
- SQL
- Docker

Example 2:
Job Description: "Needed: data scientist with Python, R, AWS, and machine learning experience."
Extracted Keywords:
- Python
- R
- AWS
- machine learning
"""

job_description = """
We are looking for a machine learning engineer with experience 
in Python, TensorFlow, Docker, and AWS. Candidates should also 
have knowledge of data pipelines, CI/CD (Jenkins), and advanced statistics.
"""

user_prompt = f"""
Now extract the keywords for the following job description:
"{job_description}"

Answer with bullet points:
"""

# Combine everything into one prompt:
prompt = f"{system_prompt}\n\n{few_shot_examples}\n\n{user_prompt}"


# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")  # or "cpu" if no GPU

# Generate output
with torch.no_grad():
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=5000,    # limit the response length
        temperature=0.2,       # low temperature = more deterministic
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True
    )

# Decode the tokens
answer = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print(answer)
