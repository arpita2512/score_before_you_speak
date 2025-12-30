import stanza
import json
from tqdm import tqdm
import sys
from pathlib import Path

def create_data(data_file):
    with open(data_file, "r", encoding="utf8") as f:
        persona =[]
        query = []
        response = []
        cand = []
        is_persona = False
        tmp_persona = []
        tmp_query = []
        tmp_response = []
        tmp_cand = []
        first = True
        cnt = 0
        sum_u = 0
        for line in f:
            cnt += 1
            line = line.strip()
            if "your persona: " in line:
                if not is_persona and not first:
                    query.append(tmp_query)
                    response.append(tmp_response)
                    cand.append(tmp_cand)
                    sum_u += len(tmp_query)
                    tmp_query = []
                    tmp_response = []
                    tmp_cand = []
                first = False
                is_persona = True
                line = line.split(": ", maxsplit=1)[1]
                tmp_persona.append(line)
            else:
                if is_persona:
                    persona.append(tmp_persona)
                    is_persona = False
                    tmp_persona = []
                line = line[line.find(" ")+1:]
                tmp_query.append(line.split("\t")[0])
                tmp_response.append(line.split("\t")[1])
                tmp_cand.append(line.split("\t")[3].split("|"))
        query.append(tmp_query)
        response.append(tmp_response)
        cand.append(tmp_cand)
        sum_u += len(tmp_query)
        assert len(query) == len(response) == len(persona) == len(cand)

    print("{} has {} dialog and {} query".format(data_file, len(query), sum_u))

    return persona, query, response, cand

def main(file_path):
    # extract data
    persona, query, response, cand = create_data(file_path)
    split = Path(file_path).parts[-1].split('_')[0]
    data = []
    for i in range(len(persona)):
        data.append({
            'persona': persona[i],
            'queries': query[i],
            'responses': response[i],
            #'distractors': cand[i]
        })

    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos', use_gpu=False, pos_batch_size=300000)

    for chat in tqdm(data):
        pos_tags= []
        for response in chat['persona']:
            res = nlp(response)
            pos_tags.append([(word.text, word.xpos) for sent in res.sentences for word in sent.words])
            chat['persona_postags'] = pos_tags
        assert len(chat['persona_postags']) == len(chat['persona']), "Check code!"
    
    with open(f'{split}_pos_tagged.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main(sys.argv[1])


    
