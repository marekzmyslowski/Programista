from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4",
                 openai_api_key="YOUR_KEY")
# Pytanie do modelu
question = "Jakie są główne zalety sztucznej inteligencji?"
# Utworzenie konwersacji z modelem
conversation = ConversationChain(llm=llm)
# Wysłanie pytania do modelu
output = conversation.run(question)
# Wyświetlenie odpowiedzi
print(output)
