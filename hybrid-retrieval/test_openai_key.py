from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, OpenAIError

load_dotenv()  # Reads OPENAI_API_KEY from .env

try:
    client = OpenAI()
    client.models.list()
    print("OpenAI API key works.")
except AuthenticationError:
    print("Invalid or unauthorized OpenAI API key.")
except OpenAIError as error:
    print(f"OpenAI API error: {error}")