import os
import openai
from dotenv import load_dotenv
from loguru import logger
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.readers import SimpleDirectoryReader
from llama_index.llms import OpenAI as LlamaOpenAI

# Load environment variables
load_dotenv()

# Set OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")

# Logging setup
logger.add("logs/app.log", rotation="10 MB")

# Define the LlamaIndex service context
service_context = ServiceContext.from_defaults(llm=LlamaOpenAI(model="gpt-4"))

# Initialize OpenAI client (New API format)
client = openai.OpenAI(api_key=api_key)

def get_nutrition_info(food_item: str) -> str:
    """
    Fetch nutrition details of a food item using the latest OpenAI API.
    """
    try:
        # Use the latest OpenAI API format
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a food nutrition expert."},
                {"role": "user", "content": f"Give detailed nutrition info for: {food_item}"}
            ]
        )
        # Extract the text from the response
        nutrition_data = response.choices[0].message.content
        logger.info(f"Fetched nutrition data for {food_item}")
        return nutrition_data
    except Exception as e:
        logger.error(f"Error fetching nutrition info: {str(e)}")
        return "Failed to retrieve nutrition details."

def build_food_index(data_path: str = "data") -> VectorStoreIndex:
    """
    Builds a food nutrition index using LlamaIndex from local text files.
    """
    try:
        documents = SimpleDirectoryReader(data_path).load_data()
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        logger.info("Food nutrition index successfully built.")
        return index
    except Exception as e:
        logger.error(f"Error building food index: {str(e)}")
        return None

# Example Usage
if __name__ == "__main__":
    food = "Apple"
    print(get_nutrition_info(food))

    # Build index from data directory (if using local text files for food data)
    food_index = build_food_index()
    if food_index:
        print("Food nutrition index successfully created!")
