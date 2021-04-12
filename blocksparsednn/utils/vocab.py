from collections import Counter
from typing import Dict, List


UNK = '<UNK>'
PAD = '<PAD>'


class Vocabulary:

    def __init__(self, max_size: int, min_count: int):
        self._max_size = max_size
        self._min_count = min_count

        self._word_counts: Counter = Counter()

        # Include the unknown and pad characters in the initial dictionary
        self._vocab: Dict[str, int] = dict()
        self._vocab[UNK] = 0
        self._vocab[PAD] = 1

        self._rev_vocab: List[str] = [UNK, PAD]

        self._is_built = False

    def add(self, word: str):
        assert not self._is_built, 'Cannot add words after the vocabulary is built'
        self._word_counts[word] += 1

    def add_multiple(self, words: List[str]):
        for w in words:
            self.add(w)

    def build(self):
        if self._is_built:
            return

        if len(self._word_counts) > self._max_size - len(self._vocab):
            top_words = self._word_counts.most_common(self._max_size)
        else:
            top_words = self._word_counts

        idx = len(self._vocab)
        for word, count in top_words.items():
            if count >= self._min_count:
                self._vocab[word] = idx
                self._rev_vocab.append(word)
                idx += 1

        self._is_built = True

    def get_id(self, word: str) -> int:
        assert self._is_built, 'Must call build() first'
        unk_id = self._vocab[UNK]
        return self._vocab.get(word, unk_id)

    def get_ids_for_seq(self, words: List[str], seq_length: int) -> List[int]:
        ids: List[int] = [self.get_id(w) for i, w in enumerate(words) if i < seq_length]

        while len(ids) < seq_length:
            ids.append(self.get_id(PAD))

        return ids

    def get_word_or_unk(self, word_id: int) -> str:
        assert self._is_built, 'Must call build() first'
        return self._rev_vocab[word_id]

    def as_dict(self) -> Dict[str, Any]:
        return {
            'vocab': self._vocab,
            'rev_vocab': self._rev_vocab,
            'max_size': self._max_size,
            'min_count': self._min_count
        }

    @classmethod
    def restore(cls, vocab_dict: Dict[str, Any]):
        vocab = cls(max_size=vocab_dict['max_size'], min_count=vocab_dict['min_count'])
        vocab._vocab = vocab_dict['vocab']
        vocab._rev_vocab = vocab_dict['rev_vocab']
        vocab._is_built = True
        return vocab
