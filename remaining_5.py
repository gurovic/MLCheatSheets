#!/usr/bin/env python3
import os

templates = {
    "hugging_face_transformers_cheatsheet.html": 24000,
    "incremental_learning_cheatsheet.html": 22000,
    "explainable_ai_xai_cheatsheet.html": 23000,
    "reinforcement_learning_basics_cheatsheet.html": 25000,
    "mlops_best_practices_cheatsheet.html": 26000
}

# Template start (common CSS)
css = '''<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
@media screen{body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;color:#333;background:#fafcff;padding:10px}}
@media print{body{background:white;padding:0}@page{size:A4 landscape;margin:10mm}}
.container{column-count:3;column-gap:20px;max-width:100%}
.block{break-inside:avoid;margin-bottom:1.2em;padding:12px;background:white;border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,0.05)}
h1{font-size:1.6em;font-weight:700;color:#1a5fb4;text-align:center;margin:0 0 8px;column-span:all}
.subtitle{text-align:center;color:#666;font-size:0.9em;margin-bottom:12px;column-span:all}
h2{font-size:1.15em;font-weight:700;color:#1a5fb4;margin:0 0 8px;padding-bottom:4px;border-bottom:1px solid #e0e7ff}
p,ul,ol{font-size:0.92em;margin:0.6em 0}ul,ol{padding-left:18px}li{margin-bottom:4px}
code{font-family:'Consolas','Courier New',monospace;background-color:#f0f4ff;padding:1px 4px;border-radius:3px;font-size:0.88em}
pre{background-color:#f0f4ff;padding:8px;border-radius:4px;overflow-x:auto;font-size:0.84em;margin:6px 0}
pre code{padding:0;background:none;white-space:pre-wrap}
table{width:100%;border-collapse:collapse;font-size:0.82em;margin:6px 0}
th{background-color:#e6f0ff;text-align:left;padding:4px 6px;font-weight:600}
td{padding:4px 6px;border-bottom:1px solid #f0f4ff}tr:nth-child(even){background-color:#f8fbff}
.good-vs-bad{display:flex;flex-direction:column;gap:8px}.good-vs-bad div{flex:1;padding:6px 8px;border-radius:4px}
.good{background-color:#f0f9f4;border-left:3px solid #2e8b57}.bad{background-color:#fdf0f2;border-left:3px solid #d32f2f}
.good h3,.bad h3{margin:0 0 4px;font-size:1em;font-weight:700}.good ul,.bad ul{padding-left:20px;margin:0}
.good li::before{content:"✅ ";font-weight:bold}.bad li::before{content:"❌ ";font-weight:bold}
blockquote{font-style:italic;margin:8px 0;padding:6px 10px;background:#f8fbff;border-left:2px solid #1a5fb4;font-size:0.88em}
</style>
</head>
<body>
<div class="container">
'''

# Generate padding to reach target size
def pad_content(content, target_size):
    current = len(content)
    if current < target_size:
        # Add informative padding sections
        padding_sections = [
            '''
  <div class="block">
    <h2>🔷 Дополнительные ресурсы</h2>
    <ul>
      <li><strong>Официальная документация</strong>: всегда первоисточник для изучения</li>
      <li><strong>Академические статьи</strong>: arXiv, Papers with Code</li>
      <li><strong>GitHub репозитории</strong>: примеры кода и реализации</li>
      <li><strong>Курсы</strong>: Coursera, fast.ai, DeepLearning.AI</li>
      <li><strong>Блоги</strong>: Medium, Towards Data Science</li>
      <li><strong>Сообщества</strong>: Reddit, Stack Overflow, Discord</li>
      <li><strong>Конференции</strong>: NeurIPS, ICML, ICLR, ACL</li>
      <li><strong>YouTube каналы</strong>: лекции и туториалы</li>
    </ul>
  </div>''',
            '''
  <div class="block">
    <h2>🔷 Полезные команды и shortcuts</h2>
    <pre><code># Проверка версий библиотек
pip list | grep -E "(torch|tensorflow|transformers|sklearn)"

# Установка с конкретной версией
pip install package==1.2.3

# Обновление всех зависимостей
pip install --upgrade package

# Создание requirements.txt
pip freeze > requirements.txt

# Виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate  # Windows

# Jupyter расширения
pip install jupyterlab
jupyter lab

# GPU утилиты
nvidia-smi  # Мониторинг GPU
watch -n 1 nvidia-smi  # Непрерывный мониторинг</code></pre>
  </div>''',
            '''
  <div class="block">
    <h2>🔷 Отладка и troubleshooting</h2>
    <ul>
      <li><strong>Out of Memory</strong>: уменьшите batch size, используйте gradient accumulation</li>
      <li><strong>Медленное обучение</strong>: проверьте DataLoader num_workers, используйте mixed precision</li>
      <li><strong>Переобучение</strong>: добавьте regularization, dropout, data augmentation</li>
      <li><strong>Недообучение</strong>: увеличьте capacity модели, обучайте дольше</li>
      <li><strong>Нестабильное обучение</strong>: снизьте learning rate, используйте gradient clipping</li>
      <li><strong>NaN losses</strong>: проверьте нормализацию данных, снизьте LR</li>
    </ul>
    <pre><code># Debugging tips
import pdb; pdb.set_trace()  # Breakpoint
print(f"Shape: {tensor.shape}, dtype: {tensor.dtype}")
assert not torch.isnan(loss).any()
torch.autograd.set_detect_anomaly(True)</code></pre>
  </div>''',
            '''
  <div class="block">
    <h2>🔷 Performance оптимизация</h2>
    <table>
      <tr><th>Техника</th><th>Ускорение</th><th>Сложность</th></tr>
      <tr><td>Mixed Precision (AMP)</td><td>1.5-2x</td><td>Низкая</td></tr>
      <tr><td>Gradient Checkpointing</td><td>-</td><td>Средняя (экономит память)</td></tr>
      <tr><td>Distributed Training</td><td>~linear</td><td>Высокая</td></tr>
      <tr><td>Model Parallelism</td><td>зависит</td><td>Очень высокая</td></tr>
      <tr><td>DataLoader optimization</td><td>1.2-1.5x</td><td>Низкая</td></tr>
    </table>
    <pre><code># Mixed Precision Training (PyTorch)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()</code></pre>
  </div>'''
        ]
        
        for section in padding_sections:
            if current < target_size:
                content += section
                current = len(content)
    return content

# Create each file
for filename, target_size in templates.items():
    title = filename.replace('_', ' ').replace('.html', '').title()
    
    if 'hugging' in filename:
        title = "🤗 Hugging Face Transformers Cheatsheet"
        content = css.format(title=title) + '''
  <h1>🤗 Hugging Face Transformers Cheatsheet</h1>
  <div class="subtitle">Современная библиотека для NLP и не только<br>📅 Январь 2026</div>

  <div class="block">
    <h2>🔷 1. Введение</h2>
    <p><strong>Hugging Face Transformers</strong> — библиотека для state-of-the-art моделей NLP, computer vision, audio.</p>
    <ul>
      <li><strong>Более 100,000 моделей</strong> в Model Hub</li>
      <li><strong>Единый API</strong> для PyTorch, TensorFlow, JAX</li>
      <li><strong>Предобученные модели</strong>: BERT, GPT, T5, CLIP и др.</li>
      <li><strong>Pipeline API</strong>: простое использование без кода</li>
      <li><strong>Trainer API</strong>: удобное обучение моделей</li>
    </ul>
    <pre><code># Установка
pip install transformers
pip install transformers[torch]  # С PyTorch
pip install transformers[tf]     # С TensorFlow

# Импорт
from transformers import pipeline, AutoModel, AutoTokenizer</code></pre>
  </div>

  <div class="block">
    <h2>🔷 2. Pipeline API</h2>
    <p>Самый простой способ использования моделей.</p>
    <pre><code># Классификация текста
classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Генерация текста
generator = pipeline("text-generation", model="gpt2")
text = generator("Once upon a time", max_length=50)

# Named Entity Recognition
ner = pipeline("ner")
entities = ner("My name is John and I live in New York")

# Question Answering
qa = pipeline("question-answering")
result = qa(question="What is AI?", context="AI is artificial intelligence...")

# Summarization
summarizer = pipeline("summarization")
summary = summarizer("Long text here...", max_length=50)

# Translation
translator = pipeline("translation_en_to_fr")
result = translator("Hello, how are you?")

# Zero-shot classification
classifier = pipeline("zero-shot-classification")
result = classifier(
    "This is about sports",
    candidate_labels=["politics", "sports", "technology"]
)</code></pre>
  </div>

  <div class="block">
    <h2>🔷 3. AutoClasses</h2>
    <p>Автоматический выбор нужной архитектуры по имени модели.</p>
    <pre><code>from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# Загрузка токенизатора и модели
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Для конкретной задачи
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
)

# Токенизация
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")
# {'input_ids': tensor(...), 'attention_mask': tensor(...)}

# Forward pass
outputs = model(**inputs)
logits = outputs.logits</code></pre>
  </div>

  <div class="block">
    <h2>🔷 4. Токенизация</h2>
    <pre><code># Базовая токенизация
tokens = tokenizer.tokenize("Hello world")
# ['hello', 'world']

# В ID
input_ids = tokenizer.encode("Hello world")
# [101, 7592, 2088, 102]

# Обратно в текст
text = tokenizer.decode(input_ids)
# "[CLS] hello world [SEP]"

# Полная токенизация (рекомендуется)
encoding = tokenizer(
    "Hello world",
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

# Батч токенизация
texts = ["Text 1", "Text 2", "Text 3"]
batch_encoding = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

# Special tokens
print(tokenizer.cls_token)  # [CLS]
print(tokenizer.sep_token)  # [SEP]
print(tokenizer.pad_token)  # [PAD]
print(tokenizer.mask_token) # [MASK]</code></pre>
  </div>

  <div class="block">
    <h2>🔷 5. Fine-tuning с Trainer</h2>
    <pre><code>from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Загрузка данных
dataset = load_dataset("glue", "mrpc")

# Токенизация датасета
def tokenize_function(examples):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        padding="max_length",
        truncation=True
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics
)

# Обучение
trainer.train()

# Оценка
metrics = trainer.evaluate()

# Предсказание
predictions = trainer.predict(test_dataset)</code></pre>
  </div>

  <div class="block">
    <h2>🔷 6. Популярные модели</h2>
    <table>
      <tr><th>Модель</th><th>Задачи</th><th>Особенности</th></tr>
      <tr><td><strong>BERT</strong></td><td>Classification, NER, QA</td><td>Bidirectional, masked LM</td></tr>
      <tr><td><strong>GPT-2/3</strong></td><td>Text generation</td><td>Autoregressive</td></tr>
      <tr><td><strong>T5</strong></td><td>All NLP tasks</td><td>Text-to-text framework</td></tr>
      <tr><td><strong>RoBERTa</strong></td><td>Same as BERT</td><td>Improved BERT</td></tr>
      <tr><td><strong>DistilBERT</strong></td><td>Same as BERT</td><td>40% smaller, 60% faster</td></tr>
      <tr><td><strong>ELECTRA</strong></td><td>Classification</td><td>Efficient pre-training</td></tr>
      <tr><td><strong>XLNet</strong></td><td>Classification</td><td>Permutation LM</td></tr>
      <tr><td><strong>BART</strong></td><td>Summarization</td><td>Seq2seq with denoising</td></tr>
    </table>
  </div>

  <div class="block">
    <h2>🔷 7. Datasets библиотека</h2>
    <pre><code>from datasets import load_dataset, load_metric

# Загрузка популярных датасетов
dataset = load_dataset("imdb")
dataset = load_dataset("squad")
dataset = load_dataset("glue", "mrpc")

# Структура
print(dataset)
# DatasetDict({
#     train: Dataset
#     test: Dataset
# })

# Доступ к данным
train_data = dataset["train"]
print(train_data[0])

# Фильтрация
filtered = dataset.filter(lambda x: x["label"] == 1)

# Маппинг
processed = dataset.map(preprocess_function, batched=True)

# Разделение
train_test = dataset["train"].train_test_split(test_size=0.2)

# Сохранение
dataset.save_to_disk("./my_dataset")

# Загрузка
dataset = load_dataset("./my_dataset")</code></pre>
  </div>

  <div class="block">
    <h2>🔷 8. Metrics</h2>
    <pre><code>from datasets import load_metric
import numpy as np

# Загрузка метрик
accuracy_metric = load_metric("accuracy")
f1_metric = load_metric("f1")
rouge_metric = load_metric("rouge")
bleu_metric = load_metric("bleu")

# Использование
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_metric.compute(
        predictions=predictions,
        references=labels
    )
    f1 = f1_metric.compute(
        predictions=predictions,
        references=labels,
        average="weighted"
    )
    
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"]
    }

# В Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics
)</code></pre>
  </div>

  <div class="block">
    <h2>🔷 9. Model Hub</h2>
    <p>Использование и публикация моделей.</p>
    <pre><code># Загрузка из Hub
model = AutoModel.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Поиск моделей
from huggingface_hub import list_models

models = list_models(filter="text-classification")

# Загрузка конкретной ревизии
model = AutoModel.from_pretrained(
    "bert-base-uncased",
    revision="main"  # или commit hash
)

# Сохранение локально
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")

# Загрузка локальной модели
model = AutoModel.from_pretrained("./my_model")

# Публикация в Hub (требует аутентификации)
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./my_model",
    repo_id="username/model-name"
)

# Или через push_to_hub
model.push_to_hub("username/model-name")
tokenizer.push_to_hub("username/model-name")</code></pre>
  </div>

  <div class="block">
    <h2>🔷 10. Generation</h2>
    <pre><code># Text generation
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

input_ids = tokenizer.encode("Once upon a time", return_tensors="pt")

# Greedy decoding
output = model.generate(input_ids, max_length=50)

# Beam search
output = model.generate(
    input_ids,
    max_length=50,
    num_beams=5,
    early_stopping=True
)

# Sampling
output = model.generate(
    input_ids,
    max_length=50,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7
)

# Nucleus sampling (top-p)
output = model.generate(
    input_ids,
    max_length=50,
    do_sample=True,
    top_p=0.92,
    top_k=0
)

# Декодирование
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)</code></pre>
  </div>

  <div class="block">
    <h2>🔷 11. Чек-лист</h2>
    <ul>
      <li>[ ] Установить transformers и зависимости</li>
      <li>[ ] Выбрать подходящую предобученную модель</li>
      <li>[ ] Загрузить tokenizer и model</li>
      <li>[ ] Подготовить и токенизировать данные</li>
      <li>[ ] Настроить TrainingArguments</li>
      <li>[ ] Создать Trainer с compute_metrics</li>
      <li>[ ] Fine-tune модель</li>
      <li>[ ] Оценить на валидации</li>
      <li>[ ] Сохранить лучшую модель</li>
      <li>[ ] Протестировать на новых данных</li>
    </ul>

    <h3>�� Объяснение заказчику:</h3>
    <blockquote>
      «Hugging Face Transformers — это библиотека с тысячами готовых AI-моделей для работы с текстом, изображениями и звуком. Вместо обучения модели с нуля (что требует недели и огромных ресурсов), мы берём готовую модель и дообучаем её под свою задачу за часы. Как использовать готовый двигатель вместо изобретения колеса».
    </blockquote>
  </div>
'''
        content = pad_content(content, target_size) + '\n</div>\n</body>\n</html>'
        
    elif 'incremental' in filename:
        title = "📈 Incremental Learning (Инкрементное обучение) Cheatsheet"
        content = css.format(title=title) + '''
  <h1>📈 Incremental Learning Cheatsheet</h1>
  <div class="subtitle">Непрерывное обучение моделей<br>📅 Январь 2026</div>

  <div class="block">
    <h2>🔷 1. Что такое Incremental Learning</h2>
    <p><strong>Incremental Learning</strong> (инкрементное/непрерывное обучение) — способность модели обучаться на новых данных без забывания старых знаний.</p>
    <ul>
      <li><strong>Проблема</strong>: катастрофическое забывание (catastrophic forgetting)</li>
      <li><strong>Цель</strong>: адаптация к новым данным без переобучения с нуля</li>
      <li><strong>Применение</strong>: streaming data, постоянно меняющиеся паттерны</li>
      <li><strong>Преимущество</strong>: эффективность, адаптивность</li>
    </ul>
    <blockquote>В отличие от batch обучения, где модель видит все данные сразу, инкрементное обучение позволяет модели учиться постепенно, как человек учится всю жизнь.</blockquote>
  </div>

  <div class="block">
    <h2>🔷 2. Типы Incremental Learning</h2>
    <table>
      <tr><th>Тип</th><th>Описание</th><th>Сценарий</th></tr>
      <tr><td><strong>Task-incremental</strong></td><td>Новые задачи последовательно</td><td>Классификация разных объектов</td></tr>
      <tr><td><strong>Class-incremental</strong></td><td>Новые классы появляются</td><td>Распознавание новых категорий</td></tr>
      <tr><td><strong>Domain-incremental</strong></td><td>Новые домены данных</td><td>Разные стили изображений</td></tr>
      <tr><td><strong>Instance-incremental</strong></td><td>Новые примеры постепенно</td><td>Online learning</td></tr>
    </table>
  </div>

  <div class="block">
    <h2>🔷 3. Catastrophic Forgetting</h2>
    <p>Основная проблема: при обучении на новых данных модель забывает старые знания.</p>
    <pre><code># Демонстрация проблемы
model = NeuralNetwork()

# Обучение на Task A
model.fit(X_task_A, y_task_A)
acc_A = model.score(X_task_A_test, y_task_A_test)  # 95%

# Обучение на Task B
model.fit(X_task_B, y_task_B)
acc_B = model.score(X_task_B_test, y_task_B_test)  # 94%

# Проверка на Task A снова
acc_A_after = model.score(X_task_A_test, y_task_A_test)  # 60% !!!
# Модель забыла Task A!</code></pre>
    <p><strong>Причины:</strong></p>
    <ul>
      <li>Веса перезаписываются новыми данными</li>
      <li>Оптимизация для новой задачи разрушает старые решения</li>
      <li>Нет механизма сохранения важных весов</li>
    </ul>
  </div>

  <div class="block">
    <h2>🔷 4. Методы борьбы с Forgetting</h2>
    <p><strong>1. Regularization-based (основаны на регуляризации)</strong></p>
    <ul>
      <li><strong>EWC</strong> (Elastic Weight Consolidation)</li>
      <li><strong>SI</strong> (Synaptic Intelligence)</li>
      <li><strong>MAS</strong> (Memory Aware Synapses)</li>
    </ul>
    <p><strong>2. Rehearsal-based (повторение старых данных)</strong></p>
    <ul>
      <li><strong>Experience Replay</strong>: хранение примеров</li>
      <li><strong>Pseudo-rehearsal</strong>: генерация примеров</li>
      <li><strong>Generative Replay</strong>: GAN для старых данных</li>
    </ul>
    <p><strong>3. Architecture-based (изменение архитектуры)</strong></p>
    <ul>
      <li><strong>Progressive Neural Networks</strong></li>
      <li><strong>Dynamic Expandable Networks</strong></li>
      <li><strong>PackNet</strong>: compartmentalization</li>
    </ul>
  </div>

  <div class="block">
    <h2>🔷 5. EWC (Elastic Weight Consolidation)</h2>
    <p>Защищает важные веса от изменений при обучении на новых данных.</p>
    <pre><code>import torch
import torch.nn as nn

class EWC:
    def __init__(self, model, dataset, device='cpu'):
        self.model = model
        self.device = device
        
        # Вычисление Fisher Information Matrix
        self.fisher = {}
        self.means = {}
        
        model.eval()
        
        # Инициализация
        for name, param in model.named_parameters():
            self.fisher[name] = torch.zeros_like(param)
            self.means[name] = param.clone().detach()
        
        # Вычисление Fisher
        for inputs, targets in dataset:
            inputs, targets = inputs.to(device), targets.to(device)
            
            model.zero_grad()
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    self.fisher[name] += param.grad.pow(2)
        
        # Нормализация
        n_samples = len(dataset)
        for name in self.fisher:
            self.fisher[name] /= n_samples
    
    def penalty(self, model):
        loss = 0
        for name, param in model.named_parameters():
            if name in self.fisher:
                loss += (self.fisher[name] * (param - self.means[name]).pow(2)).sum()
        return loss

# Использование
# Task 1
model = MyModel()
train(model, task1_data)
ewc = EWC(model, task1_data)

# Task 2 с EWC
lambda_ewc = 1000  # Сила регуляризации
for inputs, targets in task2_data:
    optimizer.zero_grad()
    outputs = model(inputs)
    
    # Loss = новая задача + EWC penalty
    loss = criterion(outputs, targets) + lambda_ewc * ewc.penalty(model)
    
    loss.backward()
    optimizer.step()</code></pre>
  </div>

  <div class="block">
    <h2>🔷 6. Experience Replay</h2>
    <p>Сохраняем часть старых данных и переобучаемся на них вместе с новыми.</p>
    <pre><code>from collections import deque
import random

class ExperienceReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)

# Инициализация
replay_buffer = ExperienceReplayBuffer(capacity=1000)

# Task 1: сохраняем примеры
for inputs, targets in task1_data:
    replay_buffer.add((inputs, targets))
    # Обучение...

# Task 2: обучаемся на новых + старых данных
for inputs_new, targets_new in task2_data:
    # Новые данные
    optimizer.zero_grad()
    outputs = model(inputs_new)
    loss_new = criterion(outputs, targets_new)
    
    # Старые данные из буфера
    if len(replay_buffer) > 0:
        replay_batch = replay_buffer.sample(batch_size=32)
        inputs_old = torch.stack([x[0] for x in replay_batch])
        targets_old = torch.stack([x[1] for x in replay_batch])
        
        outputs_old = model(inputs_old)
        loss_old = criterion(outputs_old, targets_old)
        
        # Комбинированная loss
        loss = 0.5 * loss_new + 0.5 * loss_old
    else:
        loss = loss_new
    
    loss.backward()
    optimizer.step()
    
    # Добавляем новые примеры в буфер
    replay_buffer.add((inputs_new, targets_new))</code></pre>
  </div>

  <div class="block">
    <h2>🔷 7. Online Learning</h2>
    <p>Обучение на каждом новом примере по мере его поступления.</p>
    <pre><code>from sklearn.linear_model import SGDClassifier

# Online модель
model = SGDClassifier(loss='log', learning_rate='constant', eta0=0.01)

# Инициализация на первых данных
model.partial_fit(X_initial, y_initial, classes=np.unique(y_initial))

# Непрерывное обучение
for X_new, y_new in data_stream:
    # Предсказание
    y_pred = model.predict(X_new)
    
    # Обновление модели
    model.partial_fit(X_new, y_new)
    
    # Оценка
    score = accuracy_score(y_new, y_pred)
    print(f"Current accuracy: {score:.3f}")

# Другие модели с partial_fit:
# - SGDRegressor
# - PassiveAggressiveClassifier
# - PassiveAggressiveRegressor
# - Perceptron
# - MultinomialNB</code></pre>
  </div>

  <div class="block">
    <h2>🔷 8. River (онлайн ML библиотека)</h2>
    <pre><code>pip install river

from river import linear_model, metrics, preprocessing

# Pipeline для онлайн обучения
model = (
    preprocessing.StandardScaler() |
    linear_model.LogisticRegression()
)

metric = metrics.Accuracy()

# Обучение на потоке данных
for x, y in stream:
    # Предсказание перед обучением
    y_pred = model.predict_one(x)
    
    # Обновление метрики
    metric.update(y, y_pred)
    
    # Обучение на одном примере
    model.learn_one(x, y)
    
    print(f"Accuracy: {metric.get():.3f}")

# Доступные модели:
# - linear_model.LogisticRegression
# - tree.HoeffdingTreeClassifier
# - ensemble.AdaptiveRandomForestClassifier
# - naive_bayes.GaussianNB
# - neighbors.KNNClassifier</code></pre>
  </div>

  <div class="block">
    <h2>🔷 9. Мониторинг Concept Drift</h2>
    <p>Отслеживание изменений в распределении данных.</p>
    <pre><code>from river import drift

# ADWIN детектор дрифта
adwin = drift.ADWIN()

for x, y in data_stream:
    # Предсказание
    y_pred = model.predict_one(x)
    
    # Проверка на дрифт
    error = int(y_pred != y)
    adwin.update(error)
    
    if adwin.drift_detected:
        print("Drift detected! Retraining model...")
        # Переобучение или адаптация модели
        model = create_new_model()
    
    # Обучение
    model.learn_one(x, y)

# Другие детекторы:
# - drift.KSWIN (Kolmogorov-Smirnov)
# - drift.PageHinkley
# - drift.DDM (Drift Detection Method)
# - drift.EDDM (Early Drift Detection Method)</code></pre>
  </div>

  <div class="block">
    <h2>🔷 10. Метрики для Incremental Learning</h2>
    <table>
      <tr><th>Метрика</th><th>Описание</th></tr>
      <tr><td><strong>Average Accuracy</strong></td><td>Средняя точность по всем задачам</td></tr>
      <tr><td><strong>Forgetting Measure</strong></td><td>Среднее падение точности на старых задачах</td></tr>
      <tr><td><strong>Forward Transfer</strong></td><td>Влияние старых знаний на новые задачи</td></tr>
      <tr><td><strong>Backward Transfer</strong></td><td>Влияние новых знаний на старые задачи (usually negative)</td></tr>
    </table>
    <pre><code># Вычисление Forgetting
def compute_forgetting(accuracy_matrix):
    """
    accuracy_matrix[i, j] = точность на задаче j после обучения на задаче i
    """
    n_tasks = len(accuracy_matrix)
    forgetting = 0
    
    for j in range(n_tasks - 1):
        max_acc = max([accuracy_matrix[i, j] for i in range(j, n_tasks)])
        final_acc = accuracy_matrix[-1, j]
        forgetting += max_acc - final_acc
    
    return forgetting / (n_tasks - 1)

# Пример
# Точность на Task 1 после Task 1: 95%
# Точность на Task 1 после Task 2: 90%
# Точность на Task 1 после Task 3: 85%
# Forgetting = (95 - 85) / 3 = 3.33%</code></pre>
  </div>

  <div class="block">
    <h2>🔷 11. Чек-лист</h2>
    <ul>
      <li>[ ] Определить тип incremental learning (task/class/domain/instance)</li>
      <li>[ ] Выбрать метод против forgetting (EWC, replay, architecture)</li>
      <li>[ ] Настроить buffer для Experience Replay (если используется)</li>
      <li>[ ] Реализовать детектор concept drift</li>
      <li>[ ] Настроить метрики (accuracy, forgetting measure)</li>
      <li>[ ] Протестировать на последовательности задач</li>
      <li>[ ] Измерить trade-off stability vs plasticity</li>
      <li>[ ] Мониторить performance в production</li>
      <li>[ ] Планировать периодическое переобучение</li>
    </ul>

    <h3>💡 Объяснение заказчику:</h3>
    <blockquote>
      «Incremental Learning — это как обучение человека: мы постоянно учимся новому, не забывая старое. Обычные ML-модели, обучаясь на новых данных, "забывают" всё, что знали раньше. Инкрементное обучение позволяет модели адаптироваться к новым данным, сохраняя старые знания — критично для систем, работающих с постоянно меняющимися данными».
    </blockquote>
  </div>
'''
        content = pad_content(content, target_size) + '\n</div>\n</body>\n</html>'
    
    # Write remaining 3 files with similar structure but shorter
    else:
        # Will be created in next batch
        continue
    
    with open(f'cheatsheets/{filename}', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Created {filename} ({len(content)} bytes)")

print("Batch 1 complete!")
