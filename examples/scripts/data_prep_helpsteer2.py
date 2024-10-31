import json
from transformers import AutoTokenizer


def prepare_chat_data(jsonl):
    if 'conversations' in jsonl:
        for item in jsonl['conversations']:
            if 'from' in item:
                item['role'] = item['from'].lower()
                del item['from']

            if 'value' in item:
                item['content'] = item['value']
                del item['value']

    return jsonl


def process_file(tokenizer, data_file, out_file):
    # data in pairs, assume size is even
    jsonl_data = []
    with open(data_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) % 2 == 0

        for i in range(0, len(lines)-1, 2): # in pairs
            jsonl1 = prepare_chat_data(json.loads(lines[i]))
            jsonl2 = prepare_chat_data(json.loads(lines[i+1]))

            sum1=sum2=0
            help1=help2=0
            
            for i in range(0, len(jsonl1['conversations'])):

                conversation1 = jsonl1['conversations'][i]
                conversation2 = jsonl2['conversations'][i]


                # "label": "helpfulness:3,correctness:4,coherence:4,complexity:2,verbosity:1"
                jsonl1_label = conversation1['label']
                jsonl2_label = conversation2['label']

                if jsonl1_label is None:
                    continue 

                items1 = jsonl1_label.split(',')
                items2 = jsonl2_label.split(',')

                labels1 = [int(item.split(':')[1]) for item in items1]
                labels2 = [int(item.split(':')[1]) for item in items2]  

                sum1 += sum(labels1)
                help1 += labels1[0]
                sum2 += sum(labels2)
                help2 += labels2[0]

            if help1 == help2:
                if sum1 >= sum2:
                    chosen = jsonl1
                    rejected = jsonl2
                else:
                    chosen = jsonl2
                    rejected = jsonl1
            elif sum1 >= sum2:
                chosen = jsonl1
                rejected = jsonl2
            else:
                chosen = jsonl2
                rejected = jsonl1

            new_data = chosen
            #new_data['prompt'] = chosen['conversations'][0]['content'] if chosen['conversations'][0]['role'] == 'user' else chosen['conversations'][1]['content']
            new_data['chosen'] = chosen['conversations'] #chosen['conversations'][0]['content'] if chosen['conversations'][0]['role'] == 'assistant' else chosen['conversations'][1]['content']
            new_data['rejected'] = rejected['conversations'] #rejected['conversations'][0]['content'] if  rejected['conversations'][0]['role'] == 'assistant' else rejected['conversations'][1]['content']

            jsonl_data.append(new_data)

    with open(out_file, 'w') as fout:
        for jsonl in jsonl_data:
            fout.write(json.dumps(jsonl) + '\n')

if __name__ == '__main__':
    model = '/proj/checkpoints/bathen/models/base/granite-3.0-8b-instruct'
    tokenizer = AutoTokenizer.from_pretrained(model)

    process_file(tokenizer, '/u/bathen/data/helpsteer2/train.jsonl', '/proj/checkpoints/bathen/data/helpsteer2/rm_regular/train.jsonl')
    process_file(tokenizer, '/u/bathen/data/helpsteer2/val.jsonl', '/proj/checkpoints/bathen/data/helpsteer2/rm_regular/val.jsonl') 

