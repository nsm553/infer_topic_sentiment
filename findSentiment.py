

# # To fine-tune the BERT model for sentiment analysis on a general survey response, you can use popular deep learning libraries like TensorFlow or PyTorch. Below, I'll provide a Python code example using the Transformers library by Hugging Face, which simplifies the process of working with pre-trained language models like BERT.

# # Please note that this code assumes you have already installed the required libraries. If you haven't installed them, you can do so using pip install transformers torch.

# In this code, we use a sample dataset containing survey responses and their corresponding sentiment labels. You should replace this dataset with your own labeled data. The BERT model is fine-tuned using the Trainer class from the Transformers library, which handles the training process efficiently. The TrainingArguments object sets various hyperparameters for training, like the number of epochs, batch size, etc.

# Please ensure you have enough GPU memory to fine-tune the BERT model, as it can be computationally intensive. Adjust the batch size and other hyperparameters according to your available resources and dataset size.

# After training, you can use the fine-tuned model to predict sentiment on new survey responses by passing the text through the model and interpreting the output.

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import Trainer, TrainingArguments

# Sample dataset containing survey responses and their corresponding sentiment labels
# Modify this with your own dataset
survey_responses = [
    {"text": "I really enjoyed the experience. It was fantastic!", "sentiment": 1},
    {"text": "The service was terrible. I wouldn't recommend it.", "sentiment": 0},
    # Add more data points here...
]

class SurveyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].flatten(),
            'attention_mask': encoded['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Prepare the data
    texts = [data['text'] for data in survey_responses]
    labels = [data['sentiment'] for data in survey_responses]
    dataset = SurveyDataset(texts, labels, tokenizer, max_length=128)

    # Fine-tuning parameters
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir='./logs',
        logging_steps=100,
    )

    # Create the Trainer instance and start training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

if __name__ == "__main__":
    main()
