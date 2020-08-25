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

    def __str__(self) -> str:
        return 'maxlen{}'.format(self.max_len)


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

    def __str__(self) -> str:
        return '[join_str]maxlen{}'.format(self.max_len)
    
