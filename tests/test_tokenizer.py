import unittest
from rouge.utils.tokenize import Rouge155Tokenizer

class Rouge155TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Rouge155Tokenizer()

    def test_stemmer(self):
        # WordNet Exclusion
        self.assertEqual(self.tokenizer.stem("freest"), "free",
                         "WordNet adjective exclusion failed")
        self.assertEqual(self.tokenizer.stem("hardest"), "hard",
                         "WordNet adverb exclusion failed")
        self.assertEqual(self.tokenizer.stem("candelabra"), "candelabrum",
                         "WordNet noun exclusion failed")
        self.assertEqual(self.tokenizer.stem("hobnobbing"), "hobnob",
                         "WordNet verb exclusion failed")

        # Porter Stemmer
        plurals = ['caresses', 'flies', 'dies', 'mules', 'denied',
                   'died', 'agreed', 'owned', 'humbled', 'sized',
                   'meeting', 'stating', 'siezing', 'itemization',
                   'sensational', 'traditional', 'reference', 'colonizer',
                   'plotted']
        singles = "caress fli di mule deni di agre own humbl size meet state siez item sensat tradit refer colon plot"
        stemmed_plurals = [self.tokenizer.stem(plural) for plural in plurals]
        self.assertEqual(" ".join(stemmed_plurals), singles,
                         "Porter Stemmer on plurals failed")
        # TODO: implement porter stemmer tests from nltk

    def test_stopword_remover(self):
        self.assertIsNone(self.tokenizer.remove_stopwords("afterwards"),
                          "Stopword removal failed")

    def test_sentence_splitter(self):
        spl_summary = "This is a sentence.\nFollowed by another sentence.\nAnd yet another."
        split_sentences = self.tokenizer._get_sentence_splitter("SPL")
        self.assertEqual(split_sentences(spl_summary), spl_summary.split("\n"),
                         "SPL sentence splitting failed")

        split_sentences = self.tokenizer._get_sentence_splitter(None)
        self.assertEqual(split_sentences(spl_summary), [spl_summary],
                         "None sentence splitting failed")

        see_summary = '<html>\n' \
                      '<head>\n' \
                      '<title>SL.P.10.R.A.SL062003-24</title>\n' \
                      '</head>\n' \
                      '<body bgcolor="white">\n' \
                      '<a name="1">[1]</a> <a href="#1" id=1>This is a sentence.</a>\n' \
                      '<a name="2">[2]</a> <a href="#2" id=2>Followed by another sentence.</a>\n' \
                      '<a name="3">[3]</a> <a href="#3" id=3>And yet another.</a>\n' \
                      '</body>\n</html>\n'
        split_sentences = self.tokenizer._get_sentence_splitter("SEE")
        self.assertEqual(split_sentences(see_summary), spl_summary.split("\n"),
                         "SEE sentence splitting failed")


    def test_byte_limit(self):
        text = "A testcase is created by subclassing unittest.TestCase. The three individual tests are defined with " \
               "methods whose names start with the letters test. This naming convention informs the test runner about" \
               " which methods represent tests."
        byte_limit=100
        tokenizer = Rouge155Tokenizer(stem=False, sw_removal=False, byte_limit=byte_limit)
        tokenized_text = tokenizer.tokenize_text(text)
        self.assertEqual(len("".join("".join(s) for s in tokenized_text)), byte_limit,
                         "Byte limit truncation failed")

    def test_word_limit(self):
        text = "A testcase is created by subclassing unittest.TestCase. The three individual tests are defined with " \
               "methods whose names start with the letters test. This naming convention informs the test runner about" \
               " which methods represent tests."
        word_limit=15
        tokenizer = Rouge155Tokenizer(stem=False, sw_removal=False, word_limit=word_limit)
        tokenized_text = tokenizer.tokenize_text(text)
        self.assertEqual(len(sum(tokenized_text, [])), word_limit,
                         "Word limit truncation failed")


if __name__ == '__main__':
    unittest.main()