import os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the PromptTemplate
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Tell me a short and interesting fact about {topic}."
)

# Get user input
user_input = input("Enter a topic: ")

# Format the prompt with user input
formatted_prompt = prompt.format(topic=user_input)

# Get response from LLM
response = llm(formatted_prompt)

# Print the AI-generated fact
print("\nAI Response:", response)
