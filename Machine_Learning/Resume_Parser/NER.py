import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# This is a general English NER model
# model_name = "dslim/bert-base-NER" # this model is ass
model_name = "GalalEwida/LLM-BERT-Model-Based-Skills-Extraction-from-jobdescription" # This is pretty good

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Create a NER pipeline
ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    grouped_entities=True  # group sub-word tokens into a single entity
)


# job_description = (
#     "We are looking for a machine learning engineer with experience in Python, "
#     "TensorFlow, Docker, and AWS. Candidates should also have knowledge of "
#     "data pipelines, CI/CD (Jenkins), and advanced statistics."
# )

job_description = (
    '''About the job
We have an internship opportunity working onsite at GN Jabra Intelligent Vision Systems group, located in Cupertino, CA.

This is an onsite role and the expectation is to work from our Cupertino office.

We're looking for candidates who have an excellent understanding of engineering fundamentals, specifically in the areas of embedded software and Android based software development.

In joining Jabra as a Software Intern, you will have opportunities to work on state-of-the-art/industry-leading products. This internship opportunity has the potential to develop into a full-time permanent position depending upon performance and team fit after 4-6 months.

As a member of our Software Engineering development team, you will gain valuable experience in how to develop and deploy industry-leading software products in conjunction with our industry-first PanaCast multi-camera array systems and related products.

Job Responsibilities:

Design, develop, test, and deploy industry-leading multi-camera products
Explore latest hardware and software technologies and build proof-of-concepts for next generation products
Contribute to team-based projects

Qualifications:

MS/BS student with major in Computer Science and Computer/Electrical Engineering
Strong programming skills in C/C++ on Linux, Android or RTOS
Good knowledge of operation and programming of I/O interfaces like Ethernet, USB, PCIe, SPI and I2C
Knowledge of Android Apps development and Android framework
Excellent written and oral communication skills

Pay Transparency Notice: The hourly wage for this position can range from $25.00 to $40.00. Compensation for roles at GN depend on a wide array of factors including but not limited to location, role, skill set, and level of experience.

Equal Opportunity Employer:

GN Audio/Jabra makes life sound better by developing intelligent sound solutions that transform lives through the power of sound, enabling you to hear more, do more & be more than you ever thought possible. Our integrated headset and communications solutions assist professionals in all types of businesses in being more productive. Our wireless headsets and earbuds are designed to fit any lifestyle - from sports enthusiasts to commuters and office workers. Jabra is part of the GN group, which operates in more than 90 countries across the world. Founded in 1869, GN group today has more than 6,000 employees. GN Audio an EEO Employer and does not discriminate in employment on the basis of race, color, religion, gender, national origin or ancestry, age, disability, veteran status, military service, sexual orientation, genetic information, or gender identity.

View The EEO is the Law poster and its supplement.

View the Pay Transparency Nondiscrimination Provision

E-Verify:

GN Audio / Jabra participates in E-Verify. View the E-Verify poster here. View the Right to Work poster here.

Disability Accommodation:

If you have a disability and you believe you need a reasonable accommodation in order to search for a job opening or to submit an online application, please e-mail careers.us@jabra.com or call 978-606-2210. This email and phone number is created exclusively to assist disabled job seekers whose disability prevents them from being able to apply online. Only messages left for this purpose will be returned. Messages left for other purposes, such as following up on an application or technical issues not related to a disability, will not receive a response.
'''
)

# Run the pipeline
entities = ner_pipeline(job_description)
print(entities)
