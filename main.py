from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

MODEL_NAME = "gemma3:4b"
TEMPERATURE = 0.7

llm = ChatOllama(
    model = MODEL_NAME,
    temperature = TEMPERATURE, 
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),  # System Message
    MessagesPlaceholder(variable_name="chat_history"),  # Placeholder for conversation history
    ("human", "{question}"),  # Human Message

])

chain = prompt | llm | StrOutputParser() #langchain expression language

chat_history = []

def chat(question: str) -> str:
    try:
        response = chain.invoke(
            {"question": question, "chat_history": chat_history}
        )
    except Exception as e:
        print("Error",e)

    chat_history.append(HumanMessage(content=question)),
    chat_history.append(AIMessage(content=response))
    return response

user_question = "Who is Elon Musk?"

# response = chain.invoke({"question": "What is the capital of Germany?"})  # Ai Message
# response = chain.invoke({"question": user_question})  #Ai Message


# print(response)  # Output the AI's response

if __name__ == "__main__":
    print("Welcome to the AI assistant. Type 'exit' to quit.")
    while True:
        user_question = input("Please enter your question: ")
        if user_question == "exit":
            print("Exiting the AI assistant. Goodbye!")
            break
        else:
            response = chat(user_question)
            print("Bot",response)  # Output the AI's response