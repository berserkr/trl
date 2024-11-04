import json
import os
import random

def mix_generation(files_and_sampligs_tuples, sampled_mixture_path, target_count):
    """
    Expects a list of tuples: (file abs path, sampling proportion)
    Expect sampled_mixture_path which points to file containing the final mixture
    """
    data = dict()
    sampled_data = []
    stats = dict()

    # first, normalize pct of dataset to obtain from all data...
    sampling_pct = dict()
    sampling_sum = 0

    for tups in files_and_sampligs_tuples:
        fin, sampling = tups
        sampling = int(sampling)
        sampling_sum += sampling # get total
        sampling_pct[fin] = sampling # set initial sampling

        if os.path.exists(fin):
            with open(fin, 'r') as f:
                lines = f.readlines()
                stats[fin] = len(lines) # set dataset size...
                lines = None

    for k in sampling_pct.keys(): # normalize it
        sampling_pct[k] = float(sampling_pct[k] + 1) / sampling_sum

    # sort data
    print(f'Original dict: {stats}')

    sorted_data_dict = dict(sorted(stats.items(), key=lambda item: item[1]))
    print(f'Sorted dict: {sorted_data_dict}')

    total = 0
    sampled_data = []
    for k in sorted_data_dict.keys():
        target_sample = int(sampling_pct[k] * target_count)
        print(f'Target sample {target_sample} out of {stats[k]} for {k}')

        # open once and get data
        if os.path.exists(k):
            with open(k, 'r') as f:
                lines = f.readlines()

                if target_sample <= stats[k]:
                    gotten = random.sample(lines, target_sample)
                    sampled_data += gotten
                    total += target_sample

                else: # chunk it...

                    total_consumed = 0
                    for i in range(0, target_sample, stats[k]):
                        gotten = random.sample(lines, stats[k])
                        sampled_data += gotten
                        total += stats[k]
                        total_consumed += stats[k]

                    if total_consumed < target_sample:
                        gotten = random.sample(lines, target_sample-total_consumed)
                        sampled_data += gotten
                        total += (target_sample-total_consumed)
                        total_consumed += (target_sample-total_consumed)

        stats[k] = target_sample

    pct_sum = 0
    for k in stats.keys():
        stats[k] = 100 * stats[k] / total
        pct_sum += stats[k]
        
    print(stats)
    print(f'Total: {total} with target total = {target_count} - Normalized PCT: {pct_sum}')

    with open(sampled_mixture_path, 'w') as fout:
        for data in sampled_data:

            # format the data...
            try:
                jsonl = json.loads(data)
            except:
                jsonl = data # possibly already in some json format?

            fout.write(json.dumps(jsonl) + '\n')


def generate_train_val_mixes(sampling_breakdown):
    train_mixes = []
    val_mixes = []

    for dataset in datasets:
        data_path = dataset['data_path']
        data_sampling_proportion = dataset['data_sampling_proportion']

        train_mixes.append((f'{data_path}train.jsonl', data_sampling_proportion))
        val_mixes.append((f'{data_path}val.jsonl', data_sampling_proportion))

    return train_mixes, val_mixes


if __name__ == '__main__':
    datasets = [
        {
            "data_class": "PrefDataset",
            "data_name": "HelpSteer_DPO",
            "data_path": "/proj/checkpoints/tahira/data/rm-data-msgs/gold/HelpSteer_DPO/",
            "data_sampling_proportion": 10,
            "max_input_tokens": 1536,
            "max_output_tokens": 512
        },
        {
            "data_class": "PrefDataset",
            "data_name": "gsm8k",
            "data_path": "/proj/checkpoints/tahira/data/rm-data-msgs/synthetic/gsm8k/",
            "data_sampling_proportion": 15,
            "max_input_tokens": 1536,
            "max_output_tokens": 512
        },
        {
            "data_class": "PrefDataset",
            "data_name": "safetyQA_DPO",
            "data_path": "/proj/checkpoints/tahira/data/rm-data-msgs/gold/safetyQA_DPO/",
            "data_sampling_proportion": 25,
            "max_input_tokens": 1536,
            "max_output_tokens": 512
        },
        {
            "data_class": "PrefDataset",
            "data_name": "RM_mix8x22_teacher_reason",
            "data_path": "/proj/checkpoints/tahira/data/rm-data-msgs/synthetic/RM_mix8x22_teacher_reason/",
            "data_sampling_proportion": 25,
            "max_input_tokens": 1536,
            "max_output_tokens": 512
        },
        {
            "data_class": "PrefDataset",
            "data_name": "RM_mix8x7_teacher_reason",
            "data_path": "/proj/checkpoints/tahira/data/rm-data-msgs/synthetic/RM_mix8x7_teacher_reason/",
            "data_sampling_proportion": 20,
            "max_input_tokens": 1536,
            "max_output_tokens": 512
        },
        {
            "data_class": "PrefDataset",
            "data_name": "numglue",
            "data_path": "/proj/checkpoints/tahira/data/rm-data-msgs/synthetic/numglue/",
            "data_sampling_proportion": 20,
            "max_input_tokens": 1536,
            "max_output_tokens": 512
        },
        {
            "data_class": "PrefDataset",
            "data_name": "truthy-dpo-v0.1",
            "data_path": "/proj/checkpoints/tahira/data/rm-data-msgs/gold/truthy-dpo-v0.1/",
            "data_sampling_proportion": 1,
            "max_input_tokens": 1536,
            "max_output_tokens": 512
        },
        {
            "data_class": "PrefDataset",
            "data_name": "anthropic_hh",
            "data_path": "/proj/checkpoints/tahira/data/rm-data-msgs/gold/anthropic_hh/",
            "data_sampling_proportion": 30,
            "max_input_tokens": 1536,
            "max_output_tokens": 512
        },
        {
            "data_class": "PrefDataset",
            "data_name": "Agentic-DPO-V0.1",
            "data_path": "/proj/checkpoints/tahira/data/rm-data-msgs/gold/Agentic-DPO-V0.1/",
            "data_sampling_proportion": 5,
            "max_input_tokens": 1536,
            "max_output_tokens": 512
        },
        {
            "data_class": "PrefDataset",
            "data_name": "HelpSteer2",
            "data_path": "/proj/checkpoints/shared_data/rm_data/helpsteer2_pair/",
            "data_sampling_proportion": 25,
            "max_input_tokens": 1536,
            "max_output_tokens": 512
        }
    ]
    
    # gold only...
    datasets = [dataset for dataset in datasets if 'synthetic' not in dataset['data_path']]

    train_tups, val_tups = generate_train_val_mixes(datasets)
    target_count = 300000

    base_path='/proj/checkpoints/bathen/data/rm_mixtures/golden_only'

    if not os.path.exists(base_path):
        os.makedirs(base_path)
        
    mix_generation(train_tups, f'{base_path}/train.jsonl', target_count)
    mix_generation(val_tups, f'{base_path}/val.jsonl', 0.01 * target_count)

    with open(f'{base_path}/info.json', 'w', encoding='utf-8') as f:
        json.dump(datasets, f, ensure_ascii=False, indent=4)
