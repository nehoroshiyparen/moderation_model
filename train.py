import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

def load_data():
    df = pd.read_csv('dataset/processed/combined_dataset.csv')
    df['toxic'] = df['toxic'].astype(int)
    dataset = Dataset.from_pandas(df)
    return dataset

def tokenize_data(dataset, tokenizer):
    tokenized = dataset.map(
        lambda x: tokenizer(x['text'], padding="max_length", truncation=True, max_length=256),
        batched=True
    )
    tokenized = tokenized.map(lambda x: {'labels': x['toxic']})
    return tokenized

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
    
    model.to(device)

    dataset = load_data()
    tokenized_dataset = tokenize_data(dataset, tokenizer)

    training_args = TrainingArguments(
        output_dir='./results',          # Папка для сохранения модели
        num_train_epochs=3,              # Количество эпох
        per_device_train_batch_size=16,   # Батч-сайз для тренировки
        per_device_eval_batch_size=16,    # Батч-сайз для валидации
        warmup_steps=500,                # Количество шагов для warm-up
        weight_decay=0.01,               # Регуляризация (L2)
        logging_dir='./logs',            # Папка для логов
        logging_steps=10,                # Частота логирования
        evaluation_strategy="epoch",     # Оценка модели на каждой эпохе
        save_strategy="epoch",           # Сохранение модели после каждой эпохи
        learning_rate=3e-5,              # Learning rate
        gradient_accumulation_steps=2,   # Количество шагов для накопления градиентов 
        load_best_model_at_end=True,  
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset
    )

    trainer.train()

    model.save_pretrained('models/bert-toxic-model')

if __name__ == '__main__':
    train_model()