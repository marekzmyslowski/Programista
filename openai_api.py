from openai import OpenAI

# Ustaw swój klucz API
client = OpenAI(
    api_key='YOUR_KEY')
# Zadaj pytanie modelowi GPT-4
response = client.chat.completions.create(
    model="gpt-4",  # Wybierz model GPT-4
    messages=[{"role": "user",
"content": "Jakie są główne zalety sztucznej inteligencji?"}],

    max_tokens=1024
    # Maksymalna liczba tokenów w odpowiedzi
)
# Wyświetl odpowiedź
print(response.choices[0].message.content)

