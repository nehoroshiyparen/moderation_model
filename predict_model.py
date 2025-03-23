from transformers import BertTokenizer, BertForSequenceClassification
import torch

def load_model():
    tokenizer = BertTokenizer.from_pretrained('./models/bert-toxic-model')
    model = BertForSequenceClassification.from_pretrained('./models/bert-toxic-model')
    return tokenizer, model

def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    return prediction

if __name__ == '__main__':
    tokenizer, model = load_model()
    text = 'Пример текста для проверки  модели' # Пример текста для проверки  модели
    prediction = predict(text, tokenizer, model)
    print("Toxic" if prediction == 1 else "Non-toxic")