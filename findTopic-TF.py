
# This code uses TensorFlow 2.x, TensorFlow Hub, and the BERT model provided by TensorFlow Hub. It preprocesses the text using the BERT tokenizer, creates a BERT-based classification model, and trains it on the dataset. The labels are encoded using LabelEncoder to convert them into integer IDs suitable for classification.

# Adjust the hyperparameters and fine-tuning parameters as per your requirements. After training, you can save the model and use it for topic classification on new text inputs.

# Please note that you may need to install the required dependencies, including TensorFlow, TensorFlow Hub, pandas, and scikit-learn, using pip install.

import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from official.nlp import optimization

# Sample dataset containing text and their corresponding main topics
# Modify this with your own dataset
text_data = [
    {"text": "I love to play basketball. It's my favorite sport.", "topic": "sports"},
    {"text": "The recipe for this cake is simple and delicious.", "topic": "cooking"},
    # Add more data points here...
]

# Create a DataFrame from the text_data
df = pd.DataFrame(text_data)

# Split the dataset into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Encode the labels
label_encoder = LabelEncoder()
label_encoder.fit(train_df['topic'])
num_topics = len(label_encoder.classes_)

train_df['label'] = label_encoder.transform(train_df['topic'])
val_df['label'] = label_encoder.transform(val_df['topic'])

# Load BERT module from TensorFlow Hub
bert_model_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'
bert_layer = hub.KerasLayer(bert_model_url, trainable=True)

# Preprocessing function for BERT inputs
def preprocess(texts):
    input_ids_all = []
    input_masks_all = []
    input_segments_all = []

    for text in texts:
        text = text.numpy().decode('utf-8')

        input_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            pad_to_max_length=True,
            return_attention_mask=True,
        )

        input_ids_all.append(input_dict['input_ids'])
        input_masks_all.append(input_dict['attention_mask'])
        input_segments_all.append(input_dict['token_type_ids'])

    return dict(
        input_ids=tf.convert_to_tensor(input_ids_all),
        input_mask=tf.convert_to_tensor(input_masks_all),
        input_type_ids=tf.convert_to_tensor(input_segments_all),
    )

# Create model
input_word_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='input_word_ids')
input_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='input_mask')
input_type_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='input_type_ids')

bert_outputs = bert_layer([input_word_ids, input_mask, input_type_ids])
pooled_output = bert_outputs['pooled_output']
output = tf.keras.layers.Dense(num_topics, activation='softmax')(pooled_output)

model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=output)

# Define optimizer and loss
optimizer = optimization.create_optimizer(
    initial_lr=2e-5,
    num_train_steps=len(train_df) // 32 * 10,
    num_warmup_steps=len(train_df) // 32 * 2,
    optimizer_type='adamw')

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Tokenize the texts
tokenizer = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')

# Prepare the data
train_data = tf.data.Dataset.from_tensor_slices((train_df['text'].values, train_df['label'].values))
train_data = train_data.shuffle(len(train_df)).batch(32).map(preprocess).prefetch(tf.data.AUTOTUNE)

val_data = tf.data.Dataset.from_tensor_slices((val_df['text'].values, val_df['label'].values))
val_data = val_data.batch(32).map(preprocess)

# Train the model
history = model.fit(train_data,
                    validation_data=val_data,
                    epochs=10)

# Save the model
model.save('topic_classification_bert.h5')
