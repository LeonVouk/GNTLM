import re
import wikipediaapi
import json
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
nltk.download('tokenize')
nltk.download('punkt')


# List of occupations
occupations = [
    'DJ', 'academic', 'accountant', 'actor', 'actress', 'administrator', 'ambassador', 'analyst', 'animator', 'anthropologist', 'archaeologist', 'architect',
    'archivist', 'art critic', 'art curator', 'artist', 'astrologer', 'astronaut', 'astronomer', 'athlete', 'audio engineer', 'author', 'bailiff', 'baker', 'ballerina',
    'beautician', 'biochemist', 'biologist', 'biomedical scientist', 'biotechnologist', 'blogger', 'botanist', 'broadcaster', 'business consultant', 'businessman',
    'businesswoman', 'butcher', 'calligrapher', 'caricaturist', 'cartographer', 'cartoonist', 'caterer', 'cellist', 'chef', 'choreographer', 'cinematographer',
    'civil engineer', 'clerk', 'coach', 'comedian', 'commentator', 'communications consultant', 'composer', 'computer consultant', 'computer engineer',
    'computer scientist', 'conductor', 'consultant', 'cosmetologist', 'courier', 'cryptographer', 'dance instructor', 'dancer', 'dentist', 'dermatologist', 'designer',
    'diplomat', 'director', 'doctor', 'driver', 'drummer', 'ecologist', 'economist', 'editor', 'educator', 'engineer', 'entertainer', 'entrepreneur', 'epidemiologist',
    'executive producer', 'fashion designer', 'figure skater', 'film director', 'film producer', 'firefighter', 'fisherman', 'footballer', 'genealogist', 'geologist', 'goalkeeper',
    'goldsmith', 'graphic designer', 'guitarist', 'hacker', 'handball player', 'historian', 'humanitarian', 'hydrologist', 'illustrator', 'immunologist', 'industrial designer',
    'industrial engineer', 'interpreter', 'jeweler', 'journalist', 'judge', 'lawyer', 'lecturer', 'librarian', 'linguist', 'magician', 'manager', 'management consultant',
    'mathematician', 'meteorologist', 'microbiologist', 'minister', 'movie producer', 'musician', 'music producer', 'neurologist', 'neuroscientist', 'nurse', 'oceanographer',
    'officer', 'oncologist', 'ophthalmologist', 'ornithologist', 'painter', 'palaeontologist', 'pharmacist', 'philosopher', 'photographer', 'photojournalist', 'physician',
    'physicist', 'pianist', 'pilot', 'podcaster', 'poet', 'policy advisor', 'political scientist', 'politician', 'principal', 'producer', 'production designer', 'professor',
    'programmer', 'psychologist', 'publisher', 'racing driver', 'rapper', 'record producer', 'referee', 'researcher', 'scientist', 'screenwriter', 'sculptor', 'secretary',
    'seismologist', 'singer', 'skater', 'software engineer', 'soldier', 'songwriter', 'sound designer', 'speed skater', 'statistician', 'stockbroker', 'stylist', 'surgeon',
    'surveyor', 'teacher', 'televison producer', 'theologian', 'therapist', 'translator', 'veterinarian', 'vocalist', 'volcanologist', 'wrestler', 'writer'
]

# JSON file paths
json_files = [
    "data/people_data/people_data_batch_1.json",
    "data/people_data/people_data_batch_2.json",
    "data/people_data/people_data_batch_3.json",
    "data/people_data/people_data_batch_4.json",
    "data/people_data/people_data_batch_5.json",
    "data/people_data/people_data_batch_6.json",
    "data/people_data/people_data_batch_7.json",
    "data/people_data/people_data_batch_8.json",
    "data/people_data/people_data_batch_9.json",
    "data/people_data/people_data_batch_10.json",
    "data/people_data/people_data_batch_11.json",
    "data/people_data/people_data_batch_12.json",
    "data/people_data/people_data_batch_14.json",
    "data/people_data/people_data_batch_15.json",
    "data/people_data/people_data_batch_16.json",
    "data/people_data/people_data_batch_17.json",
    "data/people_data/people_data_batch_18.json",
    "data/people_data/people_data_batch_19.json",
    "data/people_data/people_data_batch_20.json",
    "data/people_data/people_data_batch_21.json",
    "data/people_data/people_data_batch_22.json",
    "data/people_data/people_data_batch_23.json",
    "data/people_data/people_data_batch_24.json",
    "data/people_data/people_data_batch_25.json",
    "data/people_data/people_data_batch_26.json",
    "data/people_data/people_data_batch_27.json",
    "data/people_data/people_data_batch_28.json",
    "data/people_data/people_data_batch_29.json",
    "data/people_data/people_data_batch_30.json",
    "data/people_data/people_data_batch_31.json",
    "data/people_data/people_data_batch_32.json",
    "data/people_data/people_data_batch_33.json",
    "data/people_data/people_data_batch_34.json"
]

def find_occupations_in_files(occupations, json_files):
    found_occupations = []
    occupation_matches = {}

    for file_path in json_files:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)

            for person in data:
                if person.get('summary') and person.get('early life'):
                    summary = person['summary']
                    summary_sentences = sent_tokenize(summary)
                    first_sent = [sent.lower() for sent in summary_sentences][0]
                    occupations_found = []
                    for occupation in occupations:
                        if re.search(f'\s{occupation}[\s!,?.\'\"]*', first_sent):
                            found_occupations.append(occupation)
                            occupations_found.append(occupation)
                    try:
                        occs_to_add = set()
                        for occ in occupations_found:
                            rest_of_elems = [x for x in occupations_found if (occ != x and occ in x)]
                            if not rest_of_elems:
                              occs_to_add.add(occ)
                        if occs_to_add:
                            occupation_matches[person['name']] = {}
                            occupation_matches[person['name']]['occupation'] = list(occs_to_add)
                            occupation_matches[person['name']]['summary'] = person['summary']
                            occupation_matches[person['name']]['early_life'] = person['early life']
                    except IndexError:
                        continue
    return found_occupations, occupation_matches


if __name__ == '__main__':
    nltk.download('tokenize')
    nltk.download('punkt')
    # Call the function to find the occupations in the JSON files
    found_occupations, person_occupation_match = find_occupations_in_files(occupations, json_files)

    # Create a .txt file to write the results
    output_file_path = "data/match_occupations_f_multiple.txt"
    with open(output_file_path, 'w') as output_file:
        # Write the found occupations to the file
        for occupation in found_occupations:
            output_file.write(occupation + '\n')

    # Create a JSON file to write the mappings of person-occupation
    output_json_file_path = "data/person_occupation_match_f_multiple.json"
    with open(output_json_file_path, 'w') as json_f:
        json.dump(person_occupation_match, json_f, indent=4)

    print("Occupations written to:", output_file_path)
    print("Person-Occupation mappings written to:", output_json_file_path)