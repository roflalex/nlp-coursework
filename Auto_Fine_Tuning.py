"""
===========================================================
BEFORE vs AFTER FINE-TUNING (SIDE-BY-SIDE COMPARISON)
===========================================================

Task:
    Classify messages as:
        0 = QUESTION
        1 = COMPLAINT
        2 = REQUEST

Shows:
    Model predictions BEFORE and AFTER training
    on the same line for easy comparison

===========================================================
"""

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import pipeline

label_names = ["QUESTION", "COMPLAINT", "REQUEST"]

id2label = {0: "QUESTION", 1: "COMPLAINT", 2: "REQUEST"}
label2id = {"QUESTION": 0, "COMPLAINT": 1, "REQUEST": 2}


# Training data
train_texts = [
    # QUESTION
    "What time does the lecture start?",
    "Where can I find the assignment?",
    "How do I submit the lab?",
    "When is the deadline?",
    "Can you explain this topic?",

    # COMPLAINT
    "The instructions are confusing.",
    "The code does not work.",
    "I cannot open the file.",
    "The lab is too difficult.",
    "The software keeps crashing.",

    # REQUEST
    "Please send me the slides.",
    "Can you give me another example?",
    "Please extend the deadline.",
    "Can you check my code?",
    "Please upload the lab sheet."
]

train_labels = [
    0,0,0,0,0,
    1,1,1,1,1,
    2,2,2,2,2
]


# Training data
test_phrases = [
    "When do we submit the project?",
    "The notebook will not run.",
    "Please give us more examples.",
    "How do I fix this error?",
    "The instructions do not make sense.",
    "Can you upload the recording?",
    "What is the difference between BERT and GPT?",
    "The model gives the wrong answer.",
    "Please check my assignment.",
    "Where do I find the requirements file?"
]

true_labels = [0,1,2,0,1,2,0,1,2,0]


# Tokenizer 
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Dataset
dataset = Dataset.from_dict({
    "text": train_texts,
    "label": train_labels
})

dataset = dataset.map(tokenize, batched=True)



# BEFORE MODEL (UNTRAINED)
before_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

before_classifier = pipeline(
    "text-classification",
    model=before_model,
    tokenizer=tokenizer
)



#Train model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="intent_model",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    logging_steps=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

print("\n================ TRAINING =================\n")
trainer.train()


# After model
after_classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer
)


# Side-by-side comparison
print("\n==============================================================")
print("BEFORE vs AFTER COMPARISON")
print("==============================================================\n")

before_results = before_classifier(test_phrases)
after_results = after_classifier(test_phrases)

correct_before = 0
correct_after = 0

for text, b_res, a_res, true in zip(test_phrases, before_results, after_results, true_labels):

    true_label = label_names[true]

    b_label = b_res["label"]
    a_label = a_res["label"]

    b_score = b_res["score"]
    a_score = a_res["score"]

    if b_label == true_label:
        correct_before += 1

    if a_label == true_label:
        correct_after += 1

    print(f"Before: {b_label:10} ({b_score:.3f})   |   After: {a_label:10} ({a_score:.3f})   |   Text: {text}   |   True: {true_label}")
