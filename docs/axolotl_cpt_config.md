Конфигурация Axolotl (YAML) — это декларативное описание всего процесса. Фреймворк сам поднимет нужные классы PyTorch, настроит упаковку данных и раскидает вычисления по видеокартам через DeepSpeed.
Для Full Fine-Tuning мы **не указываем** блок настроек adapter: lora, благодаря чему Axolotl понимает, что нужно обновлять 100% весов базовой модели.
Сохраните этот текст в файл cpt_lfm_ru.yml:
```yaml
# 1. Базовая модель и архитектура
base_model: ./lfm-russian-lexical  # Локальный путь к модели после Lexical Initialization
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer
trust_remote_code: true            # КРИТИЧЕСКИ ВАЖНО для архитектуры LFM

load_in_8bit: false                # Никакого квантования при Full FT
load_in_4bit: false
strict: false

# 2. Рецепт данных (Data Mixing)
# Axolotl сам перемешает эти файлы. Формат pretrain означает сырой текст без "вопрос-ответ"
datasets:
  - path: json
    data_files: ./data/ru_wikipedia_cleaned.jsonl
    type: pretrain
  - path: json
    data_files: ./data/en_fineweb_anchor.jsonl
    type: pretrain
  - path: json
    data_files: ./data/code_stack_anchor.jsonl
    type: pretrain

dataset_prepared_path: ./data_prepared_cpt  # Папка для кэша токенизированных данных
val_set_size: 0                             # При CPT валидация обычно не нужна (экономим время)

# 3. Упаковка контекста (Data Packing)
sequence_len: 4096
sample_packing: true
pad_to_sequence_len: true

# 4. Гиперпараметры обучения
num_epochs: 1
micro_batch_size: 4                 # Сколько блоков по 4096 токенов влезает на ОДНУ карту
gradient_accumulation_steps: 8      # Накопление шагов для увеличения глобального батча
learning_rate: 2e-5
optimizer: adamw_torch_fused        # Fused-версия работает быстрее на NVIDIA
lr_scheduler: cosine
weight_decay: 0.1
warmup_ratio: 0.02

# 5. Точность и память
bf16: auto
fp16: false
tf32: true                          # Включает тензорные ядра на картах Ampere (RTX 30xx/A100)
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false

# 6. Логирование и сохранения
wandb_project: lfm-ru-cpt           # Интеграция с Weights & Biases для графиков
wandb_name: lfm-full-ft-run
logging_steps: 10
save_steps: 1000                    # Как часто сохранять чекпоинты (в шагах)
save_strategy: steps

# 7. Распределенное обучение
deepspeed: deepspeed_configs/zero2.json # Используем встроенный конфиг Axolotl

```
## На что обратить внимание перед запуском
 * **type: pretrain:** В Axolotl этот тип означает, что в ваших .jsonl файлах данные лежат просто в виде {"text": "Сырой текст статьи..."}. Фреймворк склеит эти тексты через EOS-токен и нарежет на блоки по 4096 токенов.
 * **micro_batch_size:** Значение 4 — это отправная точка для карты на 24 ГБ. Если обучение падает с ошибкой **CUDA Out of Memory**, снижайте это число до 2 или 1, пропорционально увеличивая gradient_accumulation_steps (чтобы их произведение оставалось прежним).
 * **deepspeed_configs/zero2.json:** Этот файл уже лежит в папке самого Axolotl. ZeRO-2 — оптимальный выбор для CPT. Он распределит состояния оптимизатора по картам, но оставит веса модели целыми.
## Как это запустить
Когда конфиг и данные готовы, процесс запуска сводится к двум командам в терминале:
**Кэширование датасета**

Запускается на процессоре (CPU). Axolotl прочитает все ваши гигабайты текста, токенизирует их, упакует в блоки по 4096 токенов и сохранит в папку ./data_prepared_cpt. Во время самого обучения видеокарты не будут простаивать в ожидании текста.**Старт распределенного обучения**

Библиотека accelerate (от Hugging Face) сама найдет все доступные видеокарты в сервере, поднимет DeepSpeed и запустит синхронизированный процесс.
По итогу в папке out/ появятся чекпоинты, и последний из них будет вашей готовой фундаментальной моделью LFM-Russian-Base. Она будет понимать русский язык не хуже родного английского, но пока еще не сможет отвечать на вопросы (будет просто продолжать текст).
