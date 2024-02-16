from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Inicjalizacja tokenizera i modelu GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
# Pytanie do modelu
input_text = "What are the advantages of AI?"
# Konwersja pytania na tokeny
input_ids = tokenizer.encode(input_text,
                             return_tensors='pt')
# Generowanie odpowiedzi przez model
output = model.generate(input_ids, max_length=100,
                        num_return_sequences=1,
                        no_repeat_ngram_size=2)
# Dekodowanie i wyświetlenie wygenerowanego tekstu
generated_text = tokenizer.decode(output[0],
                                  skip_special_tokens=True)
print(generated_text)
