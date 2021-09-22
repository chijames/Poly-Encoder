import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import random
import pickle


class SelectionDataset(Dataset):
    def __init__(self, file_path, context_transform, response_transform, concat_transform, sample_cnt=None, mode='poly'):
        self.context_transform = context_transform
        self.response_transform = response_transform
        self.concat_transform = concat_transform
        self.data_source = []
        self.mode = mode
        neg_responses = []
        with open(file_path, encoding='utf-8') as f:
            group = {
                'context': None,
                'responses': [],
                'labels': []
            }
            for line in f:
                split = line.strip('\n').split('\t')
                lbl, context, response = int(split[0]), split[1:-1], split[-1]
                if lbl == 1 and len(group['responses']) > 0:
                    self.data_source.append(group)
                    group = {
                        'context': None,
                        'responses': [],
                        'labels': []
                    }
                    if sample_cnt is not None and len(self.data_source) >= sample_cnt:
                        break
                else:
                        neg_responses.append(response)
                group['responses'].append(response)
                group['labels'].append(lbl)
                group['context'] = context
            if len(group['responses']) > 0:
                self.data_source.append(group)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, index):
        group = self.data_source[index]
        context, responses, labels = group['context'], group['responses'], group['labels']
        if self.mode == 'cross':
            transformed_text = self.concat_transform(context, responses)
            ret = transformed_text, labels
        else:
            transformed_context = self.context_transform(context)  # [token_ids],[seg_ids],[masks]
            transformed_responses = self.response_transform(responses)  # [token_ids],[seg_ids],[masks]
            ret = transformed_context, transformed_responses, labels

        return ret

    def batchify_join_str(self, batch):
        if self.mode == 'cross':
            text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch = [], [], []
            labels_batch = []
            for sample in batch:
                text_token_ids_list, text_input_masks_list, text_segment_ids_list = sample[0]

                text_token_ids_list_batch.append(text_token_ids_list)
                text_input_masks_list_batch.append(text_input_masks_list)
                text_segment_ids_list_batch.append(text_segment_ids_list)

                labels_batch.append(sample[1])

            long_tensors = [text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch]

            text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch = (
                torch.tensor(t, dtype=torch.long) for t in long_tensors)

            labels_batch = torch.tensor(labels_batch, dtype=torch.long)
            return text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch, labels_batch

        else:
            contexts_token_ids_list_batch, contexts_input_masks_list_batch, \
            responses_token_ids_list_batch, responses_input_masks_list_batch = [], [], [], []
            labels_batch = []
            for sample in batch:
                (contexts_token_ids_list, contexts_input_masks_list), (responses_token_ids_list, responses_input_masks_list) = sample[:2]

                contexts_token_ids_list_batch.append(contexts_token_ids_list)
                contexts_input_masks_list_batch.append(contexts_input_masks_list)

                responses_token_ids_list_batch.append(responses_token_ids_list)
                responses_input_masks_list_batch.append(responses_input_masks_list)

                labels_batch.append(sample[-1])

            long_tensors = [contexts_token_ids_list_batch, contexts_input_masks_list_batch,
                                            responses_token_ids_list_batch, responses_input_masks_list_batch]

            contexts_token_ids_list_batch, contexts_input_masks_list_batch, \
            responses_token_ids_list_batch, responses_input_masks_list_batch = (
                torch.tensor(t, dtype=torch.long) for t in long_tensors)

            labels_batch = torch.tensor(labels_batch, dtype=torch.long)
            return contexts_token_ids_list_batch, contexts_input_masks_list_batch, \
                          responses_token_ids_list_batch, responses_input_masks_list_batch, labels_batch
