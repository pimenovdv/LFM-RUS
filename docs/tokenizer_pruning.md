Этот этап — настоящая нейрохирургия. Мы будем оперировать двумя объектами: файлом tokenizer.json (чтобы удалить слова из оглавления) и тензорами модели на PyTorch (чтобы отрезать физические веса).
Самое важное правило прунинга: **новые ID токенов должны идти строго по порядку от 0 до N-1**. Если мы просто удалим токен [105], токен [106] должен стать [105], иначе модель выдаст ошибку IndexError при попытке обратиться к несуществующему индексу.
Вот готовый, подробно закомментированный скрипт, который решает эту задачу от начала и до конца.
**Сбор статистики**
Скрипт прогоняет ваш репрезентативный датасет через токенизатор и подсчитывает, сколько раз встретился каждый token_id.**Отбор выживших**
Мы создаем массив индексов (ID), которые нужно сохранить. В него гарантированно попадают все системные токены (BOS, EOS, PAD), а также токены, чья частота превышает заданный порог.**Хирургия Токенизатора**
Скрипт открывает внутренности Hugging Face токенизатора (tokenizer.json), удаляет редкие токены, а оставшимся переназначает ID так, чтобы они шли по порядку без «дыр».**Хирургия Матриц**
Мы создаем новые, обрезанные матрицы input_embeddings и lm_head, копируем в них веса только «выживших» токенов и заменяем слои в самой модели.
### Python скрипт (Vocab Trimming)
Для запуска вам понадобятся библиотеки torch, transformers, datasets и tqdm (для красивого прогресс-бара).
```python
import json
import torch
import torch.nn as nn
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# --- НАСТРОЙКИ ---
MODEL_NAME = "liquid-lfm-base"   # Ваша исходная модель
DATASET_PATH = "./my_mixed_data" # Ваш датасет (Англ 30% + Рус 50% + Код 20%)
MIN_FREQ = 100                   # Удаляем всё, что встретилось реже 100 раз
OUTPUT_DIR = "./lfm-pruned"      # Куда сохраним "похудевшую" модель
# -----------------

print("1. Загрузка модели и токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True)

# Загружаем датасет (в примере используем потоковое чтение, чтобы не забить ОЗУ)
# Предполагается, что в датасете есть колонка 'text'
dataset = load_dataset("json", data_files=f"{DATASET_PATH}/*.jsonl", split="train")

print("2. Подсчет частотности токенов...")
token_counts = Counter()

# Считаем токены. Для скорости берем только первые N сотен тысяч строк, 
# если датасет огромен, этого будет достаточно для статистики.
for item in tqdm(dataset.select(range(500000))):
    ids = tokenizer.encode(item["text"], add_special_tokens=False)
    token_counts.update(ids)

print("3. Отбор токенов для сохранения...")
keep_ids = []
# Обязательно сохраняем все спец-токены, даже если их частота 0
special_tokens_ids = set(tokenizer.all_special_ids)

for old_id in range(tokenizer.vocab_size):
    if old_id in special_tokens_ids or token_counts[old_id] >= MIN_FREQ:
        keep_ids.append(old_id)

new_vocab_size = len(keep_ids)
print(f"Старый словарь: {tokenizer.vocab_size}")
print(f"Новый словарь: {new_vocab_size}")
print(f"Будет удалено: {tokenizer.vocab_size - new_vocab_size} токенов.")

print("4. Хирургия токенизатора (Правка JSON)...")
# Сохраняем оригинальный токенизатор во временную папку, чтобы отредактировать его JSON
tokenizer.save_pretrained("./temp_tokenizer")

with open("./temp_tokenizer/tokenizer.json", "r", encoding="utf-8") as f:
    tok_data = json.load(f)

old_vocab = tok_data["model"]["vocab"]
# Создаем обратный словарь {id: token_string}
id_to_tok_str = {v: k for k, v in old_vocab.items()}

new_vocab = {}
new_id = 0
for old_id in keep_ids:
    tok_str = id_to_tok_str[old_id]
    new_vocab[tok_str] = new_id
    new_id += 1

# Перезаписываем словарь в JSON
tok_data["model"]["vocab"] = new_vocab

with open("./temp_tokenizer/tokenizer.json", "w", encoding="utf-8") as f:
    json.dump(tok_data, f, ensure_ascii=False, indent=2)

# Загружаем "похудевший" токенизатор
pruned_tokenizer = AutoTokenizer.from_pretrained("./temp_tokenizer")

print("5. Хирургия матриц модели (PyTorch)...")
keep_indices_tensor = torch.tensor(keep_ids, dtype=torch.long)

# 5.1 Обрезаем входные эмбеддинги
old_embeddings = model.get_input_embeddings().weight.data
new_embeddings_data = old_embeddings[keep_indices_tensor] # PyTorch позволяет нарезать матрицы массивом индексов

new_embeddings = nn.Embedding(new_vocab_size, model.config.hidden_size, dtype=torch.bfloat16)
new_embeddings.weight.data = new_embeddings_data
model.set_input_embeddings(new_embeddings)

# 5.2 Обрезаем выходную голову (Language Model Head)
old_lm_head = model.get_output_embeddings().weight.data
new_lm_head_data = old_lm_head[keep_indices_tensor]

new_lm_head = nn.Linear(model.config.hidden_size, new_vocab_size, bias=False, dtype=torch.bfloat16)
new_lm_head.weight.data = new_lm_head_data
model.set_output_embeddings(new_lm_head)

# 5.3 Обновляем конфиг
model.config.vocab_size = new_vocab_size

print("6. Сохранение финальной модели...")
model.save_pretrained(OUTPUT_DIR)
pruned_tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Успех! Обрезанная модель сохранена в {OUTPUT_DIR}.")

```
> **Важное примечание:** В некоторых архитектурах (например, в оригинальной LLaMA) размер словаря должен делиться нацело на 64 или 128 для оптимальной работы тензорных ядер (Tensor Cores) внутри видеокарты. Если new_vocab_size получится неровным (например, 45 123), скрипт можно модифицировать так, чтобы он добавлял пустые dummy-токены в конец словаря, пока число не станет кратным 64 (например, 45 184).
> 
После выполнения этого скрипта ваша модель станет легче, матрицы освободятся от балласта, и вы сможете начать добавлять в pruned_tokenizer ваши новые русские токены (используя алгоритм Lexical Initialization, который мы обсуждали ранее).
