from typing import List, Callable
import nltk
import re
from pathlib import Path
import re

class BaseTokenizer(object):
    def tokenize_text(self, text: str) -> List[List[str]]:
        """Splits text into sentences or treats entire text as sentence, splits sentences into tokens.

        :param text:    Text to be tokenized.
        :return:        List of sentences where each sentence is a list of tokens.
        """
        raise NotImplemented


class Rouge155Tokenizer(BaseTokenizer):
    def __init__(self, byte_limit: int = 0, word_limit: int = 0,
                 stem: bool = True, sw_removal: bool = True,
                 wn_exceptions_path: Path = Path(__file__).parent.parent / "data/WordNet-2.0-Exceptions",
                 stopwords_path: Path = Path(__file__).parent.parent / "data/smart_common_words.txt",
                 sentence_split: str = 'SPL'
                 ):
        self.byte_limit = byte_limit
        self.word_limit = word_limit
        self.stem = self._get_stemmer(wn_exceptions_path, stem)
        self.remove_stopwords = self._get_stopword_remover(stopwords_path, sw_removal)
        self.split_sentences = self._get_sentence_splitter(sentence_split)


    @staticmethod
    def _get_stemmer(wn_exceptions_path: Path, stem: bool) -> Callable[[str], str]:
        """Create stemming function used in ROUGE 1.5.5 (method: MorphStem)

        :param wn_exceptions_path: Path to `WordNet-2.0-Exceptions` directory in ROUGE 1.5.5 data directory
        :param stem: Whether to stem or not
        :return: Stemming function or `None` if no stemming
        """
        if not stem:
            return None
        else:
            porter_stemmer = nltk.stem.porter.PorterStemmer('MARTIN_EXTENSIONS')

            exception_paths = list(wn_exceptions_path.glob("*.exc"))
            assert(len(exception_paths)==4), f"Exception files needed: [`adj.exc`, `adv.exc`, `noun.exc`, `verb.exc`]"\
                                            f":\nException files found: {exception_paths}"

            exceptions = []
            for exception_path in exception_paths:
                with open(exception_path, 'r') as f:
                    exceptions += [line.split() for line in f.read().split("\n") if line]
            exceptions_dict = {exception[0]: exception[1] for exception in exceptions}

            return lambda x: porter_stemmer.stem(exceptions_dict[x]) if x in exceptions_dict else porter_stemmer.stem(x)

    @staticmethod
    def _get_stopword_remover(stopwords_path: Path, sw_removal: bool) -> Callable[[str], str]:
        """Create stopword remover function used in ROUGE 1.5.5 (method: createNGram)

        :param stopwords_path: Path to `smart_common_words.txt` file in ROUGE 1.5.5 data directory
        :param sw_removal: Whether to remove stop words or not
        :return: Stopword remover function or `None` if no stopword removal
        """
        if not sw_removal:
            return None
        else:
            assert(stopwords_path.exists()), f"Stopword file {stopwords_path} not found"
            with stopwords_path.open('r') as f:
                stopwords = f.read().split("\n")
            stopwords_dict = {stopword: None for stopword in stopwords}
            return lambda x: stopwords_dict[x] if x in stopwords_dict else x

    @staticmethod
    def _get_sentence_splitter(sentence_split: str) -> Callable[[str], List[str]]:
        """Create sentence splitter function used in ROUGE 1.5.5 (method: readText)

        :param sentence_split: type of sentence splitting (see ROUGE 1.5.5 for more information)
        :return: Sentence splitter function
        """
        valid_methods = [None, 'SPL', 'SEE']
        assert(sentence_split in valid_methods), f"Invalid `sentence_split`, must be in {valid_methods}"

        if not sentence_split:
            return lambda x: [x]
        elif sentence_split == "SPL":
            return lambda x: x.split("\n")
        elif sentence_split == "SEE":
            pattern = "<a size=\"[0-9]+\" name=\"[0-9]+\">\[([0-9]+)\]<\/a>\s+<a href=\"\#[0-9]+\" id=[0-9]+>([^<]+)"\
                      "|<a name=\"[0-9]+\">\[([0-9]+)\]<\/a>\s+<a href=\"\#[0-9]+\" id=[0-9]+>([^<]+)"
            return lambda x: [m[-1] for m in re.findall(pattern, x)]
        elif sentence_split == "ISI":
            raise NotImplementedError # TODO: implement ISI sentence splitting, and test
        elif sentence_split == "SIMPLE":
            raise NotImplementedError # TODO: implement SIMPLE splitting for BE, and test

    @staticmethod
    def _preprocess_text(text):
        """Applies sentence-level text preprocessing used in ROUGE 1.5.5 (method: readText)

        :param text: text to be preprocessed
        :return: preprocessed text
        """
        text = re.sub("-", " - ", text)
        text = re.sub(r"[^A-Za-z0-9\-]", " ", text)
        text = re.sub(r"^\s+", "", text)
        text = re.sub(r"\s+$", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _preprocess_word(self, word):
        """Applies word-level text preprocessing used in ROUGE 1.5.5 (stemming and stopword removal)

        :param word:
        :return: preprocessed word or `None` if removed
        """
        word = word.lower()
        if self.remove_stopwords:
            word = self.remove_stopwords(word)
        if word and not re.match("^[a-z0-9\$]", word): # condition in original createNGram method
            word = None
        if self.stem and word:
            word = self.stem(word)
        return word

    def _truncate_bytes(self, tokenized_text: List[List[str]]) -> List[List[str]]:
        """Applies byte limit truncation used in ROUGE 1.5.5 (method: readText)

        Note: one character might have a length greater than one byte

        :param tokenized_text: List of sentences where each sentence is a list of tokens
        :return: Byte-limit-truncated list of sentences where each sentence is a list of tokens
        """
        byte_len = 0
        new_tokenized_text = []
        for sentence in tokenized_text:
            new_byte_len = byte_len + len("".join(sentence).encode('utf8'))
            if new_byte_len > self.byte_limit:
                tokenized_sentence = []
                for word in sentence:
                    new_byte_len = byte_len + len(word.encode('utf8'))
                    if new_byte_len >= self.byte_limit:
                        tokenized_sentence.append(word.encode('utf8')[:(self.byte_limit-byte_len)].decode())
                        new_tokenized_text.append(tokenized_sentence)
                        byte_len = self.byte_limit
                        break
                    else:
                        tokenized_sentence.append(word)
                        byte_len = new_byte_len
            else:
                new_tokenized_text.append(sentence)
                byte_len = new_byte_len
            if byte_len == self.byte_limit:
                break
        return new_tokenized_text

    def _truncate_words(self, tokenized_text: List[List[str]]) -> List[List[str]]:
        """Applies word limit truncation used in ROUGE 1.5.5 (method: readText)

        :param tokenized_text: List of sentences where each sentence is a list of tokens
        :return: Word-limit-truncated list of sentences where each sentence is a list of tokens
        """
        word_len = 0
        new_tokenized_text = []
        for sentence in tokenized_text:
            new_word_len = word_len + len(sentence)
            if new_word_len > self.word_limit:
                tokenized_sentence = sentence[:(self.word_limit-word_len)]
                new_tokenized_text.append(tokenized_sentence)
                word_len = self.word_limit
            else:
                new_tokenized_text.append(sentence)
                word_len = new_word_len
            if word_len == self.word_limit:
                break
        return new_tokenized_text

    def tokenize_text(self, text: str) -> List[List[str]]:
        """Applies tokenization used in ROUGE 1.5.5 (method: createNGram)

        :param text: text to be tokenized
        :return: List of sentences where each sentence is a list of tokens
        """
        tokenized_text = []
        sentences = self.split_sentences(text)
        for sentence in sentences:
            sentence = self._preprocess_text(sentence)
            tokenized_sentence = [self._preprocess_word(w) for w in sentence.split() if self._preprocess_word(w)]
            tokenized_text.append(tokenized_sentence)

        if self.byte_limit:
            tokenized_text = self._truncate_bytes(tokenized_text)
        if self.word_limit:
            tokenized_text = self._truncate_words(tokenized_text)
        return tokenized_text


class SpacyTokenizer(BaseTokenizer):
    def __init__(self):
        from spacy.lang.en import English
        self.nlp = English()

    def tokenize_text(self, text: str) -> List[List[str]]:
        return [[t.text for t in self.nlp(text) if not (t.is_punct or t.is_space or t.is_bracket)]]


class SpacySentenceTokenizer(BaseTokenizer):
    def __init__(self):
        from spacy.lang.en import English
        self.nlp = English()
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def tokenize_text(self, text: str) -> List[List[str]]:
        return [[t.text for t in sent if not (t.is_punct or t.is_space or t.is_bracket)]
                for sent in self.nlp(text).sents]


