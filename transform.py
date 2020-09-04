class SelectionSequentialTransform(object):
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, texts):
        input_ids_list, segment_ids_list, input_masks_list, contexts_masks_list = [], [], [], []
        for text in texts:
            tokenized_dict = self.tokenizer.encode_plus(text, max_length=self.max_len, pad_to_max_length=True)
            input_ids, input_masks = tokenized_dict['input_ids'], tokenized_dict['attention_mask']
            assert len(input_ids) == self.max_len
            assert len(input_masks) == self.max_len
            input_ids_list.append(input_ids)
            input_masks_list.append(input_masks)

        return input_ids_list, input_masks_list


class SelectionJoinTransform(object):
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.cls_id = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.sep_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.tokenizer.add_tokens(['\n'], special_tokens=True)
        self.pad_id = 0

    def __call__(self, texts):
        # another option is to use [SEP], but here we follow the discussion at:
        # https://github.com/facebookresearch/ParlAI/issues/2306#issuecomment-599180186
        context = '\n'.join(texts)
        tokenized_dict = self.tokenizer.encode_plus(context)
        input_ids, input_masks = tokenized_dict['input_ids'], tokenized_dict['attention_mask']
        input_ids = input_ids[-self.max_len:]
        input_ids[0] = self.cls_id
        input_masks = input_masks[-self.max_len:]
        input_ids += [self.pad_id] * (self.max_len - len(input_ids))
        input_masks += [0] * (self.max_len - len(input_masks))
        assert len(input_ids) == self.max_len
        assert len(input_masks) == self.max_len

        return input_ids, input_masks
    

class SelectionConcatTransform(object):
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        # in cross encoder mode, we simply add max_contexts_length and max_response_length together to form max_len
        # this (in almost all cases) ensures all the response tokens are used and as many context tokens are used as possible
        # the intuition is that responses and the last few contexts are the most important
        self.max_len = max_len
        self.cls_id = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.sep_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.tokenizer.add_tokens(['\n'], special_tokens=True)
        self.pad_id = 0

    def __call__(self, context, responses):
        # another option is to use [SEP], but here we follow the discussion at:
        # https://github.com/facebookresearch/ParlAI/issues/2306#issuecomment-599180186
        context = '\n'.join(context)
        tokenized_dict = self.tokenizer.encode_plus(context)
        context_ids, context_masks, context_segment_ids = tokenized_dict['input_ids'], tokenized_dict['attention_mask'], tokenized_dict['token_type_ids']
        ret_input_ids = []
        ret_input_masks = []
        ret_segment_ids = []
        for response in responses:
            tokenized_dict = self.tokenizer.encode_plus(response)
            response_ids, response_masks, response_segment_ids = tokenized_dict['input_ids'], tokenized_dict['attention_mask'], tokenized_dict['token_type_ids']
            response_segment_ids = [1]*(len(response_segment_ids)-1)
            input_ids = context_ids + response_ids[1:]
            input_ids = input_ids[-self.max_len:]
            input_masks = context_masks + response_masks[1:]
            input_masks = input_masks[-self.max_len:]
            input_segment_ids = context_segment_ids + response_segment_ids
            input_segment_ids = input_segment_ids[-self.max_len:]
            input_ids[0] = self.cls_id
            input_ids += [self.pad_id] * (self.max_len - len(input_ids))
            input_masks += [0] * (self.max_len - len(input_masks))
            input_segment_ids += [0] * (self.max_len - len(input_segment_ids))
            assert len(input_ids) == self.max_len
            assert len(input_masks) == self.max_len
            assert len(input_segment_ids) == self.max_len
            ret_input_ids.append(input_ids)
            ret_input_masks.append(input_masks)
            ret_segment_ids.append(input_segment_ids)
        return ret_input_ids, ret_input_masks, ret_segment_ids
