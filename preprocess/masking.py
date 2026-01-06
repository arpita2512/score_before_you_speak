from pathlib import Path
from transformers import BartForConditionalGeneration, BartTokenizer
import random
import json
from tqdm import tqdm
import sys
import string

def main(file_path):
    # open pos tagged data
    with open(f'{file_path}', 'r') as f:
        data = json.loads(f.read())
    
    random.seed(42)
    empty_masks = []

    # create empty masks
    for chat in tqdm(data):
        for sent in chat['response_postags']:
            # filter nouns
            nouns = [word[0] for word in sent if word[1].startswith("NN")]
            if len(nouns) == 0:
                continue
            #empty_masks = []

            for noun in nouns:
                # mask single nouns
                words = ["<mask>" if word[0] == noun else word[0] for word in sent]
                # if mask is last word, add full stop to avoid punctuation mask filling
                if words[-1] == '<mask>':
                    words.append(".")
                # masked sentence
                empty_masks.append(' '.join(words))
    

    # fill masks
    device = "cuda"
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)
    tok = BartTokenizer.from_pretrained("facebook/bart-large")
    model = model.to(device)    
    filled_masks = []
    batch_size = 500

    for idx in tqdm(range(0, len(empty_masks), batch_size)):
        if idx+batch_size >= len(empty_masks):
            end_idx = len(empty_masks)
        else:
            end_idx = idx+batch_size
        batch = tok(empty_masks[idx:idx+batch_size], padding=True, return_tensors="pt").to(device)
        #print(batch['input_ids'].size())
        generated_ids = model.generate(batch["input_ids"], num_return_sequences=2, max_new_tokens=5+batch['input_ids'].size(1))
        sequences = tok.batch_decode(generated_ids, skip_special_tokens=True)
        filled_masks.extend(sequences)
    
    # combine filled masks with data
    i = 0
    for chat in tqdm(data):
        all_masked = []
        for sent in chat['response_postags']:
            nouns = [word[0] for word in sent if word[1].startswith("NN")]
            #print(len(nouns))
            if len(nouns) == 0:
                continue
            end_idx = len(nouns)*2
            regenerated = filled_masks[i:i+end_idx]

            original = chat['responses'][chat['response_postags'].index(sent)]
            sent_masked = []

            for idx in range(0, len(regenerated), 2):
                corrupted = regenerated[idx].translate(str.maketrans('', '', string.punctuation)).strip().lower()
                corrupted = ' '.join(corrupted.split())
                # if top completion is the same as original sentence or null, use 2nd completion
                if corrupted == original or corrupted == '':
                    sent_masked.append(regenerated[idx+1])
                else:
                    sent_masked.append(regenerated[idx])
            all_masked.append({
                'original': original,
                'masked': sent_masked
            })
            i = i + end_idx

        del chat['response_postags'] # postags no longer needed
        chat['aug_data'] = all_masked  
    
    # save data with filled masks
    split = Path(file_path).parts[-1].split('_')[0]
    with open(f'{split}_masked.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main(sys.argv[1])