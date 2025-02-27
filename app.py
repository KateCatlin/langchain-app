import os
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define a structured PromptTemplate
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Tell me a fascinating fact about {topic} in two sentences."
)

# Get user input
user_input = input("Enter a topic: ")

# Format the prompt with user input
formatted_prompt = prompt.format(topic=user_input)

# Generate response using LangChain
response = llm(formatted_prompt)

# Print the AI response
print("\nAI Response:", response)
