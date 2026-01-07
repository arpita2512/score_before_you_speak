from pathlib import Path
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
    
    with open(f'{split}_scores.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=3)

if __name__ == "__main__":
    main(sys.argv[1])