import os
import vertexai
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the variables
api_key = os.getenv("GEMINI_API_KEY")
project_id = os.getenv("VERTEX_PROJECT_ID")
model_endpoint = os.getenv("VERTEX_MODEL_ENDPOINT")

# Initialize and return the chat instance for fine-tuned model
def multiturn_generate_content_finetuned():
    vertexai.init(project=project_id, location="us-central1")
    model = GenerativeModel(model_endpoint)
    return model.start_chat()


# Define common generation configuration and safety settings
generation_config = {
    "max_output_tokens": 2048,
    "temperature": 1,
    "top_p": 1,
}

safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}


# Generate recipe content for a given food and its ingredients
def multiturn_generate_content_rec(food_name, ingredients):
    vertexai.init(project="opensource-406515", location="us-central1")
    model = GenerativeModel("gemini-1.5-flash-001")

    prompt = f"""
    Write a recipe for {food_name} using only the following ingredients. Include both ingredients and instructions:
    {ingredients}
    """

    # Create a new chat instance
    chat = model.start_chat()

    # Send the initial prompt to the model and return the response
    response = chat.send_message(
        [prompt],
        generation_config={
            "max_output_tokens": 8192,
            "temperature": 1,
            "top_p": 0.95,
        },
        safety_settings=safety_settings
    )

    return response
