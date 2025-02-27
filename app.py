import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_openai import OpenAI
from dotenv import load_dotenv

# Load API key
load_dotenv()
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Step 1: Generate a fun fact about a topic
fact_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Tell me a one sentence fact about {topic}."
)
fact_chain = LLMChain(llm=llm, prompt=fact_prompt)

# Step 2: Turn that fact into a poem
rhyme_prompt = PromptTemplate(
    input_variables=["fact"],
    template="Turn this fact into a 8 line poem: {fact}"
)
rhyme_chain = LLMChain(llm=llm, prompt=rhyme_prompt)

# Step 3: Title the poem
title_prompt = PromptTemplate(
    input_variables=["rhyme"],
    template="Title this poem humorously using a pun: {rhyme}"
)
title_chain = LLMChain(llm=llm, prompt=title_prompt)

# Combine all chains into a sequence
chain = SimpleSequentialChain(chains=[fact_chain, rhyme_chain, title_chain], verbose=True)

# Get user input
user_input = input("Enter a topic: ")

# Run the chain
output = chain.run(user_input)

# Print the final summary
print("\nFinal title:", output)
