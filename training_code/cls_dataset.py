

import torch
from torch.utils.data import Dataset


class CLSDataset(Dataset):
    def __init__(self, samples, tokenizer, max_input_length, device):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_length
        self.max_output_len = 0
        self.samples = []
        self.device = device

        self.bos_token_id = tokenizer.encode('<s>', add_special_tokens=False)[0]
        self.eos_token_id = tokenizer.encode('</s>', add_special_tokens=False)[0]
        self.pad_token_id = tokenizer.encode('<pad>', add_special_tokens=False)[0]

        for sample in samples:
            id_ = sample['id']
            input_ids = sample['input_tokens']
            target = None
            if sample.get('target', None) is not None:
                target = sample['target']
            self.samples.append((id_, input_ids, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        id_, input_ids, target = self.samples[index]

        input_npad = self.max_input_len - len(input_ids)
        attention_mask = torch.tensor([1] * len(input_ids) + [0] * input_npad)
        input_ids = torch.tensor(input_ids + input_npad * [self.pad_token_id])

        if target is not None:
            labels = torch.tensor(target, dtype=torch.float32).to(self.device)
        else:
            labels = []

        return {'id': id_,
                'input_ids': input_ids.to(self.device),
                'attention_mask': attention_mask.to(self.device),
                'target': labels,
                }
