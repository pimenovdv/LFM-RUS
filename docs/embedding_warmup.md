Поскольку LFM требует trust_remote_code=True, программный подход через чистый PyTorch на 100% безопасен. Методы get_input_embeddings() жестко прописаны в стандарте Hugging Face, и они сработают вне зависимости от того, как Liquid AI назвали свои переменные.
Вот готовый, боевой скрипт для Этапа 1 (Embedding Warmup). Он запускается через команду accelerate launch script.py (или deepspeed script.py).
```python
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer
)
from datasets import load_from_disk # Предполагаем, что датасет уже упакован

model_path = "./lfm-russian-lexical" # Модель после прунинга и инициализации
data_path = "./data_prepared_cpt"

print("Загрузка модели...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 # Строго bfloat16 для стабильности
)

# ==========================================
# МАГИЯ ЗАМОРОЗКИ (EMBEDDING WARMUP)
# ==========================================
print("Заморозка весов...")

# 1. Замораживаем абсолютно всё
for param in model.parameters():
    param.requires_grad = False

# 2. Размораживаем входные эмбеддинги
for param in model.get_input_embeddings().parameters():
    param.requires_grad = True

# 3. Размораживаем выходную "голову" вероятностей
for param in model.get_output_embeddings().parameters():
    param.requires_grad = True

# Проверка: выводим слои, которые будут обучаться
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"Разморожено: {trainable_params:,} из {all_params:,} параметров ({100 * trainable_params / all_params:.2f}%)")
# ==========================================

print("Загрузка данных...")
# Загружаем уже упакованный в блоки по 4096 токенов датасет 
# (его можно предварительно сделать через скрипт Axolotl или вручную)
train_dataset = load_from_disk(data_path)

# Настраиваем аргументы (Агрессивный LR, малая длительность)
training_args = TrainingArguments(
    output_dir="./out_stage1_warmup",
    overwrite_output_dir=True,
    bf16=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8, # Настройте под вашу VRAM
    learning_rate=1e-4,            # LR выше, чем при Full FT!
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_steps=1000,                # Обучаем всего 1000 шагов (или 1 эпоха на малом сабсете)
    gradient_checkpointing=True,
    logging_steps=10,
    save_steps=500,
    report_to="none"               # Или "wandb"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("Старт Этапа 1 (Прогрев)...")
trainer.train()

# Сохраняем "прогретую" модель
print("Сохранение...")
trainer.save_model("./lfm-russian-warmed")
tokenizer.save_pretrained("./lfm-russian-warmed")
print("Этап 1 успешно завершен! Модель готова к Этапу 2 (Full FT).")

```
После завершения этого скрипта вы берете папку ./lfm-russian-warmed и передаете её в Axolotl как base_model для второго этапа (где unfrozen_parameters уже убраны, и разморожены все 100% весов с низким learning_rate: 2e-5).требует
