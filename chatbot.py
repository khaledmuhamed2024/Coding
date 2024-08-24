import PyPDF2
import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import os

# Example to set read permission
os.chmod('D:/org/BrainStorming/projects/AI_lawyer_system/Laws/', 0o644)


def extract_text_from_pdf(pdf_file_path):
    text = ""
    with open(pdf_file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def preprocess_text(text):
    # Remove unnecessary spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    # Further preprocessing as needed, like removing special characters, etc.
    return text


def fine_tune_gpt2(train_text):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Tokenize the text
    tokens = tokenizer(train_text, return_tensors='pt', max_length=512, truncation=True)
    input_ids = tokens.input_ids

    # Fine-tuning settings
    training_args = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=1,
        num_train_epochs=1,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=input_ids,
    )

    trainer.train()
    return model

def chatbot_response(model, tokenizer, question):
    inputs = tokenizer.encode(question, return_tensors='pt')
    outputs = model.generate(inputs, max_length=500, do_sample=True, top_p=0.95, top_k=60)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    # Example usage
    pdf_path = "D:/org/BrainStorming/projects/AI_lawyer_system/Laws/"
    text = extract_text_from_pdf(pdf_path)
    preprocessed_text = preprocess_text(text)
    model = fine_tune_gpt2(preprocessed_text)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    while True:
        question = input("You: ")
        response = chatbot_response(model, tokenizer, question)
        print(f"Chatbot: {response}")
