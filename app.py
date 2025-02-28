import os
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import OpenAI
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import langsmith.utils as ls_utils

# Load API key
load_dotenv()
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Connect to LangSmith client
client = Client()

def push_prompt_safely(prompt_name, prompt_object):
    try:
        client.push_prompt(prompt_name, object=prompt_object)
    except ls_utils.LangSmithConflictError:
        print(f"Skipping push: {prompt_name} has not changed.")

# Step 1: Generate a fun fact about a topic
fact_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Tell me a one-sentence fact about {topic}."
)
push_prompt_safely("fun-fact-prompt", fact_prompt)
fact_chain = fact_prompt | llm

# Step 2: Turn that fact into a poem
rhyme_prompt = PromptTemplate(
    input_variables=["fact"],
    template="Turn this fact into an 8-line poem: {fact}"
)
push_prompt_safely("poem-prompt", rhyme_prompt)
rhyme_chain = rhyme_prompt | llm

# Step 3: Title the poem humorously
title_prompt = PromptTemplate(
    input_variables=["rhyme"],
    template="Title this poem humorously using a pun: {rhyme}"
)
push_prompt_safely("title-prompt", title_prompt)
title_chain = title_prompt | llm

# Combine chains sequentially
chain = (
    fact_chain
    | RunnableLambda(lambda fact: print("\nFun Fact:", fact) or {"fact": fact})
    | rhyme_chain
    | RunnableLambda(lambda rhyme: print("\nPoem:", rhyme) or {"rhyme": rhyme})
    | title_chain
    | RunnableLambda(lambda title: print("\nFinal Title:", title) or title)
)

# Get user input
user_input = input("Enter a topic: ")

# Run the chain
output = chain.invoke({"topic": user_input})
