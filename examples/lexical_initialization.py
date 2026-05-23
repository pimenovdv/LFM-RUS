import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Загружаем модель и базовый токенизатор
model_name = "liquid-lfm-base" # Замените на вашу модель
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

old_vocab_size = len(tokenizer)

# 2. Допустим, мы уже обучили новый русский токенизатор 
# и отфильтровали 10 000 новых кириллических токенов
new_ru_tokens = ["Ġкошка", "Ġсобака", "ство", "Ġнейросеть", ...] 
tokenizer.add_tokens(new_ru_tokens)
model.resize_token_embeddings(len(tokenizer))

# 3. Получаем доступ к весам
input_embeddings = model.get_input_embeddings().weight.data
output_embeddings = model.get_output_embeddings().weight.data # Голова модели

# Вычисляем дефолтный средний вектор для "непереводимых" токенов
default_mean_in = input_embeddings[:old_vocab_size].mean(dim=0)
default_mean_out = output_embeddings[:old_vocab_size].mean(dim=0)

# 4. Имитация словаря переводов (на практике здесь работает скрипт-переводчик)
# Формат: {чистый_русский_токен: английский_перевод}
translation_dict = {
    "кошка": "cat",
    "собака": "dog",
    "нейросеть": "neural network"
}

# 5. Главный цикл маппинга
for new_tok in new_ru_tokens:
    # Получаем ID нового токена в расширенном словаре
    ru_tok_id = tokenizer.convert_tokens_to_ids(new_tok)
    
    # Очищаем токен от спецсимволов токенизатора (SentencePiece / BPE)
    clean_ru_word = new_tok.replace("Ġ", "").replace(" ", "").strip()
    
    # Ищем перевод
    en_translation = translation_dict.get(clean_ru_word)
    
    if en_translation:
        # Токенизируем английский перевод СТАРЫМ словарем
        # add_special_tokens=False, чтобы не прихватить токены <s> или [CLS]
        en_tok_ids = tokenizer.encode(en_translation, add_special_tokens=False)
        
        # Защита: если переводчик выдал что-то странное
        if len(en_tok_ids) > 0:
            # Извлекаем эмбеддинги всех английских саб-токенов
            en_embeds_in = input_embeddings[en_tok_ids]
            en_embeds_out = output_embeddings[en_tok_ids]
            
            # Усредняем их
            mapped_vec_in = en_embeds_in.mean(dim=0)
            mapped_vec_out = en_embeds_out.mean(dim=0)
            
            # Копируем веса в новый русский токен
            input_embeddings[ru_tok_id] = mapped_vec_in
            output_embeddings[ru_tok_id] = mapped_vec_out
            continue # Успешно смапили, идем к следующему
            
    # Если перевода нет или это просто морфема (Fallback)
    input_embeddings[ru_tok_id] = default_mean_in
    output_embeddings[ru_tok_id] = default_mean_out

print("Lexical Initialization завершена!")
model.save_pretrained("./lfm-russian-lexical")
tokenizer.save_pretrained("./lfm-russian-lexical")
