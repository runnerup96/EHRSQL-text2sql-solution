import torch
from torch.utils.data import Dataset


class T5FinetuneDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.tokenizer = tokenizer
        self.max_input_len = 0
        self.max_output_len = 0
        self.samples = []

        self.bos_token_id = tokenizer.encode('<s>', add_special_tokens=False)[0]
        self.eos_token_id = tokenizer.encode('</s>', add_special_tokens=False)[0]
        self.pad_token_id = tokenizer.encode('<pad>', add_special_tokens=False)[0]

        for sample in samples:
            id_ = sample['id']
            input_ids = sample['source_tokens']
            if sample['target_tokens']:
                output_ids = sample['target_tokens'] + [self.eos_token_id]
                self.max_output_len = max(self.max_output_len, len(output_ids))
            else:
                output_ids = None

            self.max_input_len = max(self.max_input_len, len(input_ids))
            self.samples.append((id_, input_ids, output_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        id_, input_ids, output_ids = self.samples[index]

        input_npad = self.max_input_len - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * input_npad
        input_ids = torch.LongTensor(input_ids + input_npad * [self.pad_token_id])

        if output_ids:
            output_npad = self.max_output_len - len(output_ids)
            labels = torch.LongTensor(output_ids + output_npad * [-100])
        else:
            labels = []

        return {'id': id_,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                }
