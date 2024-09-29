import os
import google.generativeai as genai
import streamlit as st
import json


from RecipeBot import *

# Create the model configuration and safety settings
generation_config = generation_config
safety_settings = safety_settings

# Generate content using the fine-tuned model
chat_ingredients = multiturn_generate_content_finetuned()

# Input field for the food name
food_name = st.text_input('Enter the name of the food you want to look up:')
if food_name:
    response = chat_ingredients.send_message(
        ["Food:{}".format(food_name)],
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    # Extracting the generated content
    ingeredients = response.candidates[0].content.text
    st.write(f"Macros: {ingeredients}")

    # Recipe generation
    res = multiturn_generate_content_rec(food_name, ingeredients)

    # Display the macronutrient information
    # st.subheader(f"Simple recipe to make: {food_name}")
    st.write(res.candidates[0].content.text)

    # Add a horizontal divider between different food items
    st.markdown("---")
