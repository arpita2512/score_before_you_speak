import json
import sys
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer

def concatenate_and_tokenize(example, tokenizer, vocab):
    example['input_ids'] = [vocab['<|startoftext|>']]
    example['input_ids'].extend(tokenizer(example['persona'])['input_ids'])
    for i in range(len(example['history'])):
        if i%2 == 0:
            example['input_ids'].append(vocab['<|sp1|>'])
        else:
           example['input_ids'].append(vocab['<|sp2|>'])
        example['input_ids'].extend(tokenizer(example['history'][i])['input_ids'])

    example['input_ids'].extend(tokenizer("Score: "+str(example['score']))['input_ids'])
    example['input_ids'].append(vocab['<|sp2|>'])
    example['input_ids'].extend(tokenizer("Bot: "+example['response'])['input_ids'])
    example['input_ids'].append(vocab['<|endoftext|>'])
    return example

def dpgt_dataset(data): 
    personas, histories, responses, scores = [], [], [], []

    # convert to HF dataset 
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
    
    ds = Dataset.from_dict({
        'persona': personas,
        'history': histories,
        'response': responses,
        'score': scores
    })

    # tokenize
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    vocab = tokenizer.get_vocab()
    special_tokens = {
        'bos_token': '<|startoftext|>',
        'additional_special_tokens': ['<|sp1|>', '<|sp2|>']
    }
    _ = tokenizer.add_special_tokens(special_tokens)
    vocab = tokenizer.get_vocab()

    kwargs = {
      "tokenizer": tokenizer,
      "vocab": vocab
    }
    processed_train_data = ds.map(
      concatenate_and_tokenize,
      remove_columns=ds.column_names,
      num_proc=4,
      fn_kwargs=kwargs
    )

    # save to disk
    processed_train_data.save_to_disk("dgpt_train_final")

def llama_dataset(data):
    personas, histories, responses, scores = [], [], [], []

    # modify to llama format (2nd person)
    for chat in tqdm(data):
        persona = ""
        for p in chat['persona']:
            p = p.replace("i'm", "you are")
            p = p.replace("i'll", "you'll")
            p = p.replace("i am ", "you are ")
            p = p.replace("i was", "you were")
            p = p.replace("i've", "you have")
            p = p.replace("my", "your")
            p = p.replace("i ", "you ")
            p = p.replace(" me ", " you ")
            persona += p + " "
        personas.extend([persona]*len(chat['responses']))
        full_dialog = [x for xs in zip(chat['queries'], chat['responses']) for x in xs]
        for resp in chat['responses']:
            idx = full_dialog.index(resp)
            hist = []
            temp = full_dialog[:idx] 
            for i in range(len(temp)):
                if i%2 == 0:
                    hist.append(temp[i])
                else:
                    hist.append(temp[i])
            histories.append(hist)
        responses.extend(chat['responses'])
        scores.extend([1.0]*len(chat['responses']))
        for aug in chat['aug_data']:
            idx = responses.index(aug['original'])
            for masked in aug['masked']:
                personas.append(personas[idx])
                histories.append(histories[idx])
                responses.append(masked['sent'])
                scores.append(masked['score'])
        assert len(personas) == len(responses) == len(histories) == len(scores), "Check code"

    outputs = []
    for idx in tqdm(range(len(responses))):
        messages = []
        messages.append({
            "role": "system",
            "content": personas[idx]
        })
        for i in range(len(histories[idx])):
            if i == (len(histories[idx]) - 1):
                messages.append({
                    'role': 'user',
                    'content': histories[idx][i] + " Score: " + str(scores[idx])
            })
            elif i%2 == 0:
                messages.append({
                    'role': 'user',
                    'content': histories[idx][i]
                })
            else:
                messages.append({
                    'role': 'assistant',
                    'content': histories[idx][i]
                })
        messages.append({
            'role': 'assistant',
            'content': responses[idx]
        })
        outputs.append(messages)
    
    # apply chat template and tokenize
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    all_inputs = []
    for dialog in tqdm(messages):
        dialog_tokens = tokenizer.apply_chat_template(dialog)
        all_inputs.append(dialog_tokens)

    processed_data = Dataset.from_dict({
        'input_ids': all_inputs
    })

    # save to disk
    processed_data.save_to_disk("llama_train_final")

def main(file_path, model_name):
    # open data
    with open(f'{file_path}', 'r') as f:
        data = json.loads(f.read())
    
    if model_name == "dgpt":
        dpgt_dataset(data)
    elif model_name == "llama":
        llama_dataset(data)
    else:
        print("Invalid model name!")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])