import json
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str)
args = parser.parse_args()

with open('ubuntu_train_subtask_1_augmented.json') as infile:
    data = json.load(infile)

outfile = open('train.txt'.format(args.mode), 'w')
for idx, content in enumerate(data):
    messages = content['messages-so-far']
    options = content['options-for-correct-answers']
    option = options[0]
    context = []
    for message in messages:
        text = message['speaker'].replace('_', ' ') + ': ' + message['utterance']
        context.append(text)
    outfile.write('{}\t{}\t{}\n'.format(1, '\t'.join(context), option['utterance'].strip()))
