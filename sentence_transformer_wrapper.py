from sentence_transformers import SentenceTransformer
from typing import Iterable
from torch import nn

from transformers import BertTokenizer
import os
import torch

CACHE_DIR = os.environ['MODEL_CACHE']


class SentenceTransformerWrapper(SentenceTransformer):
    def __init__(self, model_name_or_path: str = None, modules: Iterable[nn.Module] = None, device: str = None, num_bert_layers: int = 1):
        super().__init__(model_name_or_path, modules, device)
        self.num_bert_layers = num_bert_layers
        print("num_Bert_layers: {}".format(num_bert_layers))

    def encode_tensor(self, input_ids_tensor, attention_mask_tensor):
        token_type_ids = None
        input_dict = {"input_ids": input_ids_tensor, "attention_mask": attention_mask_tensor, "token_type_ids": token_type_ids}
        out_features = self.forward(input_dict)
        last_layer_sent_embeddings = out_features["sentence_embedding"]  # [num_sents * num_cands, 768]

        if self.num_bert_layers == 1:
            return last_layer_sent_embeddings
        else:
            all_layer_embeddings = out_features['all_layer_embeddings']  # word embeddings
            concated_embeddings = torch.cat(all_layer_embeddings[-2:-self.num_bert_layers - 1:-1], dim=-1)
            # [num_sents * num_cands, sent_len, 768 * (num_bert_layers-1)]
            input_mask_expanded = attention_mask_tensor.unsqueeze(-1).expand(concated_embeddings.size()).float()
            sum_mask = input_mask_expanded.sum(1)
            lower_layers_sent_embeddings = torch.sum(concated_embeddings, dim=1) / sum_mask
            # [num_sents * num_cands, 768 * (num_bert_layers-1)]
            concated_sent_embeddings = torch.cat([last_layer_sent_embeddings, lower_layers_sent_embeddings], dim=-1)
            return concated_sent_embeddings  # [num_sents * num_cands, 768 * num_bert_layers]


    def encode_word(self, input_ids_tensor, attention_mask_tensor):
        token_type_ids = None
        input_dict = {"input_ids": input_ids_tensor, "attention_mask": attention_mask_tensor, "token_type_ids": token_type_ids}
        out_features = self.forward(input_dict)
        #token_embeddings = out_features["token_embeddings"]
        #input_mask = out_features['attention_mask']
        all_layer_embeddings = out_features['all_layer_embeddings']
        concated_embeddings = torch.cat(all_layer_embeddings[-1:-self.num_bert_layers - 1:-1], dim=-1)
        # [num_sents * num_cands, sent_len, 768 * num_bert_layers]

        input_mask_expanded = attention_mask_tensor.unsqueeze(-1).expand(concated_embeddings.size()).float()
        embeddings = concated_embeddings * input_mask_expanded

        return embeddings



if __name__ == "__main__":
    device = "cuda:0"
    sentence_encoder = SentenceTransformerWrapper(model_name_or_path='bert-base-nli-mean-tokens', num_bert_layers=1).cuda()
    sentences = ['This framework generates embeddings for each input sentence',
                 'Sentences are passed as a list of string.',
                 'The quick brown fox jumps over the lazy dog.']
    #sentence_embeddings = sentence_encoder.encode(sentences, convert_to_tensor=True)

    # tokenize
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir=CACHE_DIR)
    tokenized_sentences = [bert_tokenizer.tokenize(t) for t in sentences]
    sentences_ids = [[101] + bert_tokenizer.convert_tokens_to_ids(sentence) + [102] for sentence in tokenized_sentences]
    max_len = max([len(sentence) for sentence in sentences_ids])
    sentences_ids_padded = []
    for sentence in sentences_ids:
        pad_len = max_len - len(sentence)
        sentences_ids_padded.append( sentence + [0] * pad_len )

    sentences_ids_tensor = torch.LongTensor(sentences_ids_padded).cuda()

    attention_mask = torch.ne(sentences_ids_tensor, 0).cuda()

    with torch.no_grad():
        word_embeddings = sentence_encoder.encode_word(sentences_ids_tensor, attention_mask)

    """
    with torch.no_grad():
        sentence_embeddings_own = sentence_encoder.encode_tensor(sentences_ids_tensor, attention_mask)
    """
