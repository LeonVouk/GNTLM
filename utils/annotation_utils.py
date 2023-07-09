import json
import glob

JSON_FILE = 'data/person_occupation_match_f_multiple.json'


def extract_annotated_data():
    f_data = []
    annotated_json_files = glob.glob(r'data/annotations/*.json')
    for jf_name in annotated_json_files:
        print(jf_name)
        with open(jf_name, 'r') as json_f:
            f = json.load(json_f)
        for person in f:
            for k, v in person.items():
              if '<<<' in v['early_life']:
                  f_data.append(person)
    return f_data


def get_people_to_annotate(amount, start_at: int=0, author: str=''):
    r_d = {}
    with open(JSON_FILE, 'r') as c_f:
        person_dict = json.load(c_f)
    people = [{_p: {'occupation': person_dict[_p]['occupation'], 'early_life': person_dict[_p]['early_life']}} for _p in list(person_dict.keys())[start_at: start_at + amount]]
    for _p in people:
        r_d.update(_p)
    try:
        with open(author + '_annotation_' + JSON_FILE, 'r') as n_f:
            json_dict = json.load(n_f)
        for i in people:
            json_dict.append(i)
    except FileNotFoundError:
        json_dict = people
    with open(author + '_annotation_' + JSON_FILE, 'w') as n_f:
        json.dump(json_dict, n_f, indent=4)


if __name__ == '__main__':
    get_people_to_annotate(amount=20, start_at=0, author='x')