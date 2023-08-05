
import tensorflow as tf
import tensorflow_hub as hub
from official.nlp import optimization
from sklearn.model_selection import train_test_split

# Sample dataset containing survey responses and their corresponding sentiment labels
# Modify this with your own dataset
survey_responses = [
    {"text": "I really enjoyed the experience. It was fantastic!", "sentiment": 1},
    {"text": "The service was terrible. I wouldn't recommend it.", "sentiment": 0},
    # Add more data points here...
]

# Split the dataset into train and validation sets
train_data, val_data = train_test_split(survey_responses, test_size=0.2, random_state=42)

# Tokenize the texts using BERT tokenizer
bert_model_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'
bert_layer = hub.KerasLayer(bert_model_url, trainable=True)

# Preprocess function for BERT inputs
def preprocess(texts):
    input_dict = tokenizer(texts, return_tensors='tf', padding=True, truncation=True, max_length=128)
    return {
        'input_word_ids': input_dict['input_ids'],
        'input_mask': input_dict['attention_mask'],
        'input_type_ids': input_dict['token_type_ids'],
    }

# Create BERT sentiment analysis model
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
inputs = preprocess(text_input)
outputs = bert_layer(inputs)
pooled_output = outputs['pooled_output']
sentiment_output = tf.keras.layers.Dense(1, activation='sigmoid', name='sentiment')(pooled_output)

model = tf.keras.Model(inputs=text_input, outputs=sentiment_output)

# Define optimizer and loss
loss = tf.keras.losses.BinaryCrossentropy()
metrics = tf.keras.metrics.BinaryAccuracy()

epochs = 3
steps_per_epoch = len(train_data) // batch_size

num_train_steps = steps_per_epoch * epochs
num_warmup_steps = num_train_steps // 10

init_lr = 2e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Prepare the data
train_texts = [data['text'] for data in train_data]
train_labels = [data['sentiment'] for data in train_data]

val_texts = [data['text'] for data in val_data]
val_labels = [data['sentiment'] for data in val_data]

batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((train_texts, train_labels)).shuffle(len(train_data)).batch(batch_size).prefetch(tf.data.AUTOTUNE).map(preprocess)
val_dataset = tf.data.Dataset.from_tensor_slices((val_texts, val_labels)).batch(batch_size).map(preprocess)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

# Save the model
model.save('sentiment_analysis_bert.h5')
