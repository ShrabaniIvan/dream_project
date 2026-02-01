import os
import requests
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables from .env
load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if the keys are loaded correctly
if openai_api_key: 
    print(f"OPENAI API key exists and begins with : {openai_api_key[:10]}...")
else:
    print("Failed to load openai_api_key.")



#use case 
with open("test.txt", "r", encoding="utf-8") as f:
    query = f.read()


messages = [{"role":"user", "content":query}]

#OpenAI part 
#Connect to OpenAI client library 
openai = OpenAI(api_key=openai_api_key)
response_openai = openai.chat.completions.create(messages=messages,model="gpt-4.1-mini")
print("OpenAI response:")
print(response_openai.choices[0].message.content)









