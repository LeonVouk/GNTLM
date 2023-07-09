import re
import wikipediaapi
import json
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
nltk.download('tokenize')
nltk.download('punkt')

from utils.corpus_utils import extract_early_life




# Create a Wikipedia API object
wiki_wiki = wikipediaapi.Wikipedia('en')

# Read the JSON file
with open('data/people_data/people_slugs.json') as file:
    data = json.load(file)

# Extract the values from the JSON data and update the 'people_names' list
people_names = data

# Create an empty list to store the extracted data
extracted_data = []

# Define the batch size
batch_size = 1000

# Iterate over the person names
for i, name in enumerate(people_names, 1):
    try:
        # Get the Wikipedia page for the person
        page = wiki_wiki.page(name)

        # Check if the page exists
        if page.exists():
            # Extract the "Early life" section
            early_life_section = extract_early_life(page)

            # Create a dictionary to store the extracted data for each person
            person_data = {
                'name': name,
                'summary': page.summary,
                'early life': early_life_section
            }

            # Append the person's data to the extracted_data list
            extracted_data.append(person_data)
    except json.JSONDecodeError:
        print(f"Error: Failed to retrieve data for {name}. Skipping.")

    # Check if it's time to save the data
    if i % batch_size == 0 or i == len(people_names):
        # Save the extracted data to a new JSON file
        batch_number = (i - 1) // batch_size + 1
        filename = f"data/people_data/people_data_batch_{batch_number}.json"
        with open(filename, 'w') as file:
            json.dump(extracted_data, file, indent=4)
        print(f"Data saved to {filename}")

        # Clear the extracted_data list for the next batch
        extracted_data = []

# Print a completion message
print("All data saved.")