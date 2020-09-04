import json
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str)
args = parser.parse_args()

with open('{}.json'.format(args.mode)) as infile:
    data = json.load(infile)

outfile = open('{}.txt'.format(args.mode), 'w')
for content in data:
    messages = content['messages-so-far']
    options = content['options-for-correct-answers']
    option = options[0]
    correct_id = option['candidate-id']
    context = []
    for message in messages:
        text = message['speaker'].replace('_', ' ') + ': ' + message['utterance']
        context.append(text)
    outfile.write('{}\t{}\t{}\n'.format(1, '\t'.join(context), option['utterance'].strip()))
    # write negative samples
    if args.mode != 'train':
        negs = content['options-for-next']
        for neg in negs:
            if neg['candidate-id'] != correct_id:
                outfile.write('{}\t{}\t{}\n'.format(0, '\t'.join(context), neg['utterance'].strip()))
    if args.mode == 'train':
        negs = content['options-for-next']
        cnt = 0
        for neg in negs:
            if neg['candidate-id'] != correct_id:
                cnt += 1
                outfile.write('{}\t{}\t{}\n'.format(0, '\t'.join(context), neg['utterance'].strip()))
                if cnt == 15: # 16-1
                    break

