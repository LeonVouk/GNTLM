import json
from utils.annotation_utils import extract_annotated_data


if __name__ == '__main__':
    extracted_data = extract_annotated_data()
    final_data = []
    for person in extracted_data:
        temp_dict = {}
        for _k, _v in person.items():
            evidence = []
            early_life = _v['early_life']
            e_spl = early_life.split('<<<')
            for i in e_spl:
                e = i.split('>>>')[0].replace('\n', ' ')
                evidence.append(e)
            temp_dict[_k] = {}
            early_life = _v['early_life'].replace('<<<', '').replace('>>>', '')
            occupation = _v['occupation']
            temp_dict[_k]['early_life'] = early_life
            temp_dict[_k]['evidence'] = evidence
            temp_dict[_k]['occupation'] = occupation
        final_data.append(temp_dict)

    with open('data/annotated_data_aggr.json', 'w') as json_f:
        json.dump(final_data, json_f, indent=4)
