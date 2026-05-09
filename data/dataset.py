import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from torch.utils.data import random_split

import datasets
from datasets import load_dataset
from preparation.tokenizer import get_or_build_tokenizer

def get_datasets(config):
    raw_data = load_dataset(path="IPSAN/tatar_translation_dataset", split="train")

    tokenizer_src = get_or_build_tokenizer(config, raw_data, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, raw_data, config["lang_tgt"])

    #split
    train_size = int(0.9 * len(raw_data))
    val_size = len(raw_data) - train_size
    train_data, val_data = random_split(raw_data, [train_size, val_size])

    train_subset_indices = list(range(8))
    val_subset_indices = list(range(8))

    train_ds = TranslationData(train_data, tokenizer_src=tokenizer_src, tokenizer_tgt=tokenizer_tgt, src_lang=config["lang_src"], tgt_lang=config["lang_tgt"], seq_len=config["seq_len"])
    val_ds = TranslationData(val_data, tokenizer_src=tokenizer_src, tokenizer_tgt=tokenizer_tgt, src_lang=config["lang_src"], tgt_lang=config["lang_tgt"], seq_len=config["seq_len"])

    train_ds = Subset(train_ds, train_subset_indices)
    val_ds = Subset(val_ds, val_subset_indices)

    max_len_src = 0
    max_len_tgt = 0

    #convert sentences into token ids to find max sequence length
    for item in raw_data:
        src_ids = tokenizer_src.encode(item[config["lang_src"]]).ids
        tgt_ids = tokenizer_src.encode(item[config["lang_tgt"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max src seq len: {max_len_src}, Max tgt seq len {max_len_tgt}")
    
    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

class TranslationData(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
    
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        src_tgt_pair = self.dataset[index]
        src_txt = src_tgt_pair[self.src_lang]
        tgt_txt = src_tgt_pair[self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_txt).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_txt).ids
        
        #calculate how many tokens we need to add to input tokens to match the sequence lengths
        # - 2 cuz we also adding the sos and eos tokens
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        # same for target sequence, -1 because we are only adding the sos token to the decoder side (because the eos token will be predicted)
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence too long")

        #add SOS, EOS and enough padding to have all items in a batch be the same dimensions
        encoder_input = torch.cat(
            [
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        #what we expect as output from the decoder 
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label": label,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # prevent real tokens from attending to padding tokens
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # ensures causal relationship in decoder input
            "src_txt": src_txt,
            "tgt_txt": tgt_txt
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0

