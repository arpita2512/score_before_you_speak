from pathlib import Path
from datasets import Dataset
import json
from tqdm import tqdm
import sys
from bert_score import BERTScorer

def make_ref_cand_pairs(data):
    all_refs, all_cands = [], []
    for chat in data:
        refs, cands = [], []
        for aug in chat['aug_data']:
            refs.extend([aug['original']]*len(aug['masked']))
            cands.extend(aug['masked'])
        assert len(refs) == len(cands), "Check code!"
        all_refs.append(refs)
        all_cands.append(cands)
    
    assert len(all_refs) == len(all_cands), "Check code!"
    return all_refs, all_cands

def main(file_path):
    split = Path(file_path).parts[-1].split('_')[0]

    # open masked data
    with open(f'{file_path}', 'r') as f:
        data = json.loads(f.read())
    
    # create references and candidates for bertscore
    refs, cands = make_ref_cand_pairs(data)

    scorer = BERTScorer(
        lang="en", rescale_with_baseline=True,
        model_type="microsoft/deberta-xlarge-mnli"
    )

    # get scores
    fscores = []
    for i in tqdm(range(len(refs))):
        P, R, F1 = scorer.score(cands[i], refs[i])
        fscores.append(F1.tolist())
    
    # add scores to data
    for i in range(len(data)):
        scores = fscores[i]
        j = 0
        for aug in data[i]['aug_data']:
            masked_with_scores = []
            for masked_sent in aug['masked']:
                masked_with_scores.append({
                    'sent': masked_sent,
                    'score': round(scores[j], 2) 
                })
                j += 1
            aug['masked'] = masked_with_scores
    
    # convert to HF dataset 
    personas, histories, responses, scores = [], [], [], []

    if split == "train":
        for chat in tqdm(data):
            persona = "Your persona:"
            for p in chat['persona']:
                persona += " " + p
            personas.extend([persona]*len(chat['responses']))
            full_dialog = [x for xs in zip(chat['queries'], chat['responses']) for x in xs]
            for resp in chat['responses']:
                idx = full_dialog.index(resp)
                hist = []
                temp = full_dialog[:idx] 
                for i in range(len(temp)):
                    if i%2 == 0:
                        hist.append("User: "+temp[i])
                    else:
                        hist.append("Bot: "+temp[i])
                histories.append(hist)
            responses.extend(chat['responses'])
            scores.extend([1.0]*len(chat['responses']))

            if len(chat['aug_data']) == 0:
                continue

            for aug in chat['aug_data']:
                idx = responses.index(aug['original'])
                for masked in aug['masked']:
                    personas.append(personas[idx])
                    histories.append(histories[idx])
                    responses.append(masked['sent'])
                    scores.append(masked['score'])
            assert len(personas) == len(responses) == len(histories) == len(scores), "Check code"
    
    else: # val or test split
        for chat in tqdm(data):
            persona = "Your persona:"
            for p in chat['persona']:
                persona += " " + p
            personas.extend([persona]*len(chat['responses']))
            full_dialog = [x for xs in zip(chat['queries'], chat['responses']) for x in xs]
            for resp in chat['responses']:
                idx = full_dialog.index(resp)
                hist = []
                temp = full_dialog[:idx] 
                for i in range(len(temp)):
                    if i%2 == 0:
                        hist.append("User: "+temp[i])
                    else:
                        hist.append("Bot: "+temp[i])
                histories.append(hist)
            responses.extend(chat['responses'])
            scores.extend([1.0]*len(chat['responses']))
            assert len(personas) == len(responses) == len(histories) == len(scores), "Check code"
    
    ds = Dataset.from_dict({
        'persona': personas,
        'history': histories,
        'response': responses,
        'score': scores
    })

    # save to disk
    ds.save_to_disk(f"{split}_final")

if __name__ == "__main__":
    main(sys.argv[1])