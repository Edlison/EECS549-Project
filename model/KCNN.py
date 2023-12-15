import torch
import torch.nn as nn
import torch.nn.functional as F
from project.model.additive import AdditiveAttention
import logging
from project.model.bert import BertModel
from project.model.tokenization import BertTokenizer
from typing import List

CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"

logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class KCNN(torch.nn.Module):
    """
    Knowledge-aware CNN (KCNN) based on Kim CNN.
    Input a news sentence (e.g. its title), produce its embedding vector.
    """

    def __init__(self, config):
        super(KCNN, self).__init__()
        self.config = config
        self.embedder = BertEmbedder()
        self.word_embedding = nn.Embedding(config.num_words,
                                           config.word_embedding_dim,
                                           padding_idx=0)
        self.entity_embedding = nn.Embedding(config.num_entities,
                                             config.entity_embedding_dim,
                                             padding_idx=0)
        self.transform_matrix = nn.Parameter(
            torch.empty(self.config.entity_embedding_dim,
                        self.config.word_embedding_dim).uniform_(-0.1, 0.1))
        self.transform_bias = nn.Parameter(
            torch.empty(self.config.word_embedding_dim).uniform_(-0.1, 0.1))

        self.conv_filters = nn.ModuleDict({
            str(x): nn.Conv2d(3 if self.config.use_context else 2,
                              self.config.num_filters,
                              (x, self.config.word_embedding_dim))
            for x in self.config.window_sizes
        })
        self.additive_attention = AdditiveAttention(
            self.config.query_vector_dim, self.config.num_filters)

    def forward(self, news):
        """
        Args:
          news:
            {
                "title": batch_size * num_words_title,
                "title_entities": batch_size * num_words_title
            }

        Returns:
            final_vector: batch_size, len(window_sizes) * num_filters
        """
        # batch_size, num_words_title, word_embedding_dim
        # word_vector = self.embedder.get_bert_embeddings(news["title"].to(device))
        word_vector = self.word_embedding(news["title"].to(device))
        # batch_size, num_words_title, entity_embedding_dim
        # entity_vector = self.embedder.get_bert_embeddings(news["title_entities"].to(device))
        entity_vector = self.entity_embedding(news["title_entities"].to(device))
        if self.config.use_context:
            # batch_size, num_words_title, entity_embedding_dim
            context_vector = self.embedder.get_bert_embeddings(
                news["title_entities"].to(device))

        # batch_size, num_words_title, word_embedding_dim
        transformed_entity_vector = torch.tanh(
            torch.add(torch.matmul(entity_vector, self.transform_matrix),
                      self.transform_bias))

        if self.config.use_context:
            # batch_size, num_words_title, word_embedding_dim
            transformed_context_vector = torch.tanh(
                torch.add(torch.matmul(context_vector, self.transform_matrix),
                          self.transform_bias))

            # batch_size, 3, num_words_title, word_embedding_dim
            multi_channel_vector = torch.stack([
                word_vector, transformed_entity_vector,
                transformed_context_vector
            ],
                dim=1)
        else:
            # batch_size, 2, num_words_title, word_embedding_dim
            multi_channel_vector = torch.stack(
                [word_vector, transformed_entity_vector], dim=1)

        pooled_vectors = []
        for x in self.config.window_sizes:
            # batch_size, num_filters, num_words_title + 1 - x
            convoluted = self.conv_filters[str(x)](
                multi_channel_vector).squeeze(dim=3)
            # batch_size, num_filters, num_words_title + 1 - x
            activated = F.relu(convoluted)
            # batch_size, num_filters
            # Here we use a additive attention module
            # instead of pooling in the paper
            pooled = self.additive_attention(activated.transpose(1, 2))
            # pooled = activated.max(dim=-1)[0]
            # # or
            # # pooled = F.max_pool1d(activated, activated.size(2)).squeeze(dim=2)
            pooled_vectors.append(pooled)
        # batch_size, len(window_sizes) * num_filters
        final_vector = torch.cat(pooled_vectors, dim=1)
        return final_vector


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def select_field(features, field):
    """As the output is dic, return relevant field"""
    return [[choice[field] for choice in feature.choices_features] for feature in features]


def create_examples(_list, set_type="train"):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(_list):
        guid = "%s-%s" % (set_type, i)
        text_a = line
        # text_b = line[1]
        examples.append(
            InputExample(guid=guid, text_a=text_a))
    return examples


class BertEmbedder:
    def __init__(self,
                 pretrained_weights='bert-base-uncased',
                 tokenizer_class=BertTokenizer,
                 model_class=BertModel,
                 max_seq_len=20):
        super().__init__()
        self.pretrained_weights = pretrained_weights
        self.tokenizer_class = tokenizer_class
        self.model_class = model_class
        self.tokenizer = self.tokenizer_class.from_pretrained(pretrained_weights)
        self.model = self.model_class.from_pretrained(pretrained_weights)
        self.max_seq_len = max_seq_len
        # tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        # model = BertModel.from_pretrained(pretrained_weights)

    def get_bert_embeddings(self, raw_text: List[str]) -> torch.tensor:
        examples = create_examples(raw_text)

        features = convert_examples_to_features(
            examples, self.tokenizer, self.max_seq_len, True)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        last_hidden_states = self.model(all_input_ids)[0]  # Models outputs are now tuples
        print(len(last_hidden_states))
        return last_hidden_states


if __name__ == "__main__":
    embedder = BertEmbedder()
    raw_text = ["[CLS] This is first element [SEP] continuing statement",
                "[CLS] second element of the list."]
    bert_embedding = embedder.get_bert_embeddings(raw_text)
