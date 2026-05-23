import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, MarianMTModel, MarianTokenizer
from typing import List
from tqdm import tqdm

def initialize_lexical_embeddings(
    model_name: str,
    new_tokens: List[str],
    translation_model_name: str = "Helsinki-NLP/opus-mt-ru-en",
    save_path: str = "./lfm-russian-lexical"
):
    """
    Implements full lexical initialization (cross-lingual word embedding alignment).
    Uses a translation model to translate new tokens to English and copies the mean
    of English token embeddings to the new token embeddings.
    """
    # 1. Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    old_vocab_size = len(tokenizer)

    # 2. Add new tokens and resize embeddings
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # 3. Access embeddings weights
    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data

    # Calculate default mean vector for "untranslatable" tokens
    default_mean_in = input_embeddings[:old_vocab_size].mean(dim=0)
    default_mean_out = output_embeddings[:old_vocab_size].mean(dim=0)

    # 4. Load translation model
    print(f"Loading translation model {translation_model_name}...")
    trans_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
    trans_model = MarianMTModel.from_pretrained(translation_model_name)

    def translate(texts: List[str]) -> List[str]:
        if not texts: return []
        inputs = trans_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        translated = trans_model.generate(**inputs)
        return [trans_tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    # 5. Main mapping loop
    print("Starting lexical mapping...")

    # Batch size for translation
    batch_size = 32
    for i in tqdm(range(0, len(new_tokens), batch_size)):
        batch_tokens = new_tokens[i:i+batch_size]

        # Clean tokens (remove BPE artifacts)
        clean_words = [tok.replace("Ġ", "").replace(" ", "").strip() for tok in batch_tokens]

        # Filter empty strings
        to_translate = []
        to_translate_indices = []
        for idx, word in enumerate(clean_words):
            if word:
                to_translate.append(word)
                to_translate_indices.append(idx)

        translations = []
        if to_translate:
             translations = translate(to_translate)

        # Process each token in the batch
        trans_idx = 0
        for idx, new_tok in enumerate(batch_tokens):
            ru_tok_id = tokenizer.convert_tokens_to_ids(new_tok)
            success = False

            if idx in to_translate_indices:
                en_translation = translations[trans_idx]
                trans_idx += 1

                if en_translation:
                    # Tokenize english translation with BASE dictionary
                    en_tok_ids = tokenizer.encode(en_translation, add_special_tokens=False)

                    if len(en_tok_ids) > 0:
                        en_embeds_in = input_embeddings[en_tok_ids]
                        en_embeds_out = output_embeddings[en_tok_ids]

                        mapped_vec_in = en_embeds_in.mean(dim=0)
                        mapped_vec_out = en_embeds_out.mean(dim=0)

                        input_embeddings[ru_tok_id] = mapped_vec_in
                        output_embeddings[ru_tok_id] = mapped_vec_out
                        success = True

            # Fallback for empty strings, untranslatable morphs, or translation failure
            if not success:
                input_embeddings[ru_tok_id] = default_mean_in
                output_embeddings[ru_tok_id] = default_mean_out

    print("Lexical Initialization completed!")
    if save_path:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    return model, tokenizer

if __name__ == "__main__":
    # Example for quick test
    initialize_lexical_embeddings(
        model_name="gpt2",
        new_tokens=["Ġкошка", "Ġсобака", "ство", "Ġнейросеть"],
        save_path="./test_out"
    )
