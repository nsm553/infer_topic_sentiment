
# To fine-tune the BERT model for determining the main topic from text, you can use a similar approach as shown in the previous example for sentiment analysis. We'll use the Transformers library by Hugging Face and PyTorch as the deep learning framework.

# Please note that the code provided here is just a basic example to get you started. Fine-tuning BERT for topic classification may require more complex data preprocessing and hyperparameter tuning, depending on your specific dataset and task.

# In this code, we assume a sample dataset containing text and their corresponding main topics. You should replace this dataset with your own labeled data. We convert the topic labels into integer IDs so that we can use them for training the BERT model.

# The rest of the code is similar to the sentiment analysis example. We fine-tune the BERT model using the Trainer class and set various hyperparameters using the TrainingArguments object.

# Once the model is fine-tuned, you can use it to predict the main topic of new text by passing the text through the model and mapping the predicted topic IDs back to their corresponding labels using the topic_id2label mapping.

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import Trainer, TrainingArguments

# Sample dataset containing text and their corresponding main topics
# Modify this with your own dataset
text_data = [
    {"text": "I love to play basketball. It's my favorite sport.", "topic": "sports"},
    {"text": "The recipe for this cake is simple and delicious.", "topic": "cooking"},
    # Add more data points here...
]

class TopicDataset(Dataset):
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
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_topics)

    # Prepare the data
    texts = [data['text'] for data in text_data]
    topic_labels = [data['topic'] for data in text_data]

    # Create a mapping of unique topic labels to integers
    unique_topics = list(set(topic_labels))
    topic_label2id = {label: i for i, label in enumerate(unique_topics)}
    topic_id2label = {i: label for i, label in enumerate(unique_topics)}

    # Convert topic labels to corresponding integer IDs
    labels = [topic_label2id[label] for label in topic_labels]
    num_topics = len(unique_topics)

    dataset = TopicDataset(texts, labels, tokenizer, max_length=128)

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
