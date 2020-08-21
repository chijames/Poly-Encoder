import json
with open('ubuntu_test_subtask_1.json') as infile:
    data = json.load(infile)

with open('ubuntu_responses_subtask_1.tsv') as infile:
    for line, content in zip(infile, data):
        _, candidate_id, utterance = line.split('\t')
        content['options-for-correct-answers'] = [{'candidate-id':candidate_id, 'utterance':utterance}]

with open('test.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)
