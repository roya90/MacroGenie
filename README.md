# MacroGenie

MacroGenie is an intelligent application designed to provide macronutrient breakdowns for various foods and meals. Whether you're tracking your diet, planning meals, or simply curious about the nutritional content of your favorite dishes, MacroGenie has you covered!

## Features

- **Food and Meal Analysis**: Input food items or recipes to receive a detailed macronutrient breakdown (Carbohydrates, Protein, and Fat).
- **Custom Dishes**: Create and analyze your own dishes by combining multiple ingredients.
- **User-Friendly Interface**: Simple and intuitive UI for easy navigation and use.

## Purpose

This project is a toy example designed to fine-tune the Gemini model in Vertex AI and demonstrate the utilization of the fine-tuned model in a practical application. It showcases the capabilities of Gemini when it is fine-tuned to process and analyze nutritional data effectively.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/roya90/MacroGenie.git
   cd MacroGenie
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

## Usage

1. Launch the application by running the `app.py` script.
2. Enter the name of a food item or select a predefined dish.
3. View the detailed macronutrient breakdown for the selected item or dish.
4. Use the "Custom Dish" feature to create and analyze your own recipes by combining ingredients.

## Project Structure

- **`MacroGenie.py`**: Main application script that orchestrates the functionality of the project.
- **`RecipeBot.py`**: Contains logic for recipe processing and analysis.
- **`Data/`**: Stores data files or the database containing nutritional information for foods and recipes to fine-tune Gemini.
- **`.gitignore`**: Specifies files and directories to be ignored by Git version control.
- **`requirements.txt`**: Lists the dependencies required to run the project.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Open a Pull Request and describe your changes.

## License

This project is licensed under the MIT License. 

## Support

If you encounter any issues or have questions, feel free to open an issue in the repository or contact the author.
