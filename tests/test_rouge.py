import unittest
from rouge.rouge import Rouge, RougeScores
from rouge.utils.tokenize import Rouge155Tokenizer
from pathlib import Path
from tqdm import tqdm

from typing import Dict

SEE_sentence_splitter = Rouge155Tokenizer._get_sentence_splitter("SEE")
SPL_sentence_splitter = Rouge155Tokenizer._get_sentence_splitter("SPL")

class PerlScriptUtils:
    """Helper functions to deal with inputs/outputs from perl ROUGE.

    """

    @staticmethod
    def parse_rouge_perl_out(rouge_output) -> Dict[str,Dict[str, RougeScores]]:
        """ Parse rouge perl script output (ROUGE-N)

        Script command to generate output:
        $ tests/perl-rouge/ROUGE-1.5.5.pl -e tests/perl-rouge/data -c 95
          -r 1000 -n 4 -x -a tests/duc2005_subset/rouge_perl.in > rouge_perl.out

        :param rouge_output: path to perl script output file
        :return: Nested dict of form {system_id: {topic_id: RougeScores}
        """
        print("Parsing ROUGE scores...")
        with open(rouge_output, "r") as f:
            out = f.read().split("\n")
        results = {}
        for row in out:
            r = row.split()
            # skip separator rows
            if len(r) != 7:
                continue
            if r[1][6] == "O":
                continue
            # parse values from row
            metric = r[1]
            system_id = r[3].split(".")[-1]
            topic_id = r[3].split(".")[0]
            R = r[4].split(":")[1]
            P = r[5].split(":")[1]
            F = r[6].split(":")[1]
            if system_id not in results:
                results[system_id] = {}
            if topic_id not in results[system_id]:
                results[system_id][topic_id] = {}
            results[system_id][topic_id][metric] = {"R": float(R),
                                                    "P": float(P),
                                                    "F": float(F)}
        return results

    @staticmethod
    def _parse_systems(systems_path: Path) -> Dict[str, Dict[str, str]]:
        """ Parse system summaries from DUC 2005.

        :param systems_path: path to directory containing system summaries
        :return: Nested dict of form {system_id: {topic_id: summary}}
        """
        systems = {}
        for path in systems_path.glob("*"):
            text = path.read_text(encoding = "ISO-8859-1")
            summary = " ".join(SPL_sentence_splitter(text))
            topic_id = str(path).split("/")[-1].split(".")[0]
            system_id = str(path).split("/")[-1].split(".")[-1]
            if system_id not in systems:
                systems[system_id] = {}
            systems[system_id][topic_id] = summary
        return systems

    @staticmethod
    def _parse_models(models_path: Path) -> Dict[str, Dict[str, str]]:
        """ Parse model summaries from DUC 2005.

        :param models_path: path to directory containing model summaries
        :return: Nested dict of form {topic_id: {model_id: summary}}
        """
        models = {}
        for path in models_path.glob("*"):
            text = path.read_text(encoding = "ISO-8859-1")
            summary = " ".join(SPL_sentence_splitter(text))
            topic_id = str(path).split("/")[-1].split(".")[0]
            model_id = str(path).split("/")[-1].split(".")[-1]
            if topic_id  not in models:
                models[topic_id] = {}
            models[topic_id][model_id] = summary
        return models


# TODO: clean up tests
class RougeTests(unittest.TestCase):
    def setUp(self):
        self.rouge = Rouge.from_rouge155_args()
        self.models_path = Path("duc2005_subset/models")
        self.systems_path = Path("duc2005_subset/peers")

    def test_n_score(self):
        models = PerlScriptUtils._parse_models(self.models_path)
        systems = PerlScriptUtils._parse_systems(self.systems_path)

        # no swr, no stem
        n_scores = {}
        for system_id, cand_texts in tqdm(systems.items()):
            if system_id not in n_scores:
                n_scores[system_id] = {}
            for topic_id, cand_text in cand_texts.items():
                if topic_id not in n_scores[system_id]:
                    n_scores[system_id][topic_id] = {}
                ref_texts = models[topic_id].values()
                n_score = self.rouge.n_score(ref_texts, cand_text)
                n_scores[system_id][topic_id] = n_score

        n_scores_perl = PerlScriptUtils.parse_rouge_perl_out("duc2005_subset/rouge_perl.out")
        for system_id, pyrouge_topics in n_scores.items():
            for topic_id, pyrouge_scores in pyrouge_topics.items():
                scores_perl = n_scores_perl[system_id][topic_id]
                for n, scores in scores_perl.items():
                    for k,v in scores.items():
                        # ROUGE truncates, while we round.
                        self.assertAlmostEqual(v, pyrouge_scores[n][k], 4,
                            "Results different from original ROUGE.")
        # swr, no stem
        self.rouge = Rouge.from_rouge155_args({"s": True})
        n_scores = {}
        for system_id, cand_texts in tqdm(systems.items()):
            if system_id not in n_scores:
                n_scores[system_id] = {}
            for topic_id, cand_text in cand_texts.items():
                if topic_id not in n_scores[system_id]:
                    n_scores[system_id][topic_id] = {}
                ref_texts = models[topic_id].values()
                n_score = self.rouge.n_score(ref_texts, cand_text)
                n_scores[system_id][topic_id] = n_score

        n_scores_perl = PerlScriptUtils.parse_rouge_perl_out("duc2005_subset/rouge_perl_swr.out")
        for system_id, pyrouge_topics in n_scores.items():
            for topic_id, pyrouge_scores in pyrouge_topics.items():
                scores_perl = n_scores_perl[system_id][topic_id]
                for n, scores in scores_perl.items():
                    for k,v in scores.items():
                        # ROUGE truncates, while we round.
                        self.assertAlmostEqual(v, pyrouge_scores[n][k], 4,
                            "Results different from original ROUGE (swr).")

        # stem, no swr
        self.rouge = Rouge.from_rouge155_args({"m": True})
        n_scores = {}
        for system_id, cand_texts in tqdm(systems.items()):
            if system_id not in n_scores:
                n_scores[system_id] = {}
            for topic_id, cand_text in cand_texts.items():
                if topic_id not in n_scores[system_id]:
                    n_scores[system_id][topic_id] = {}
                ref_texts = models[topic_id].values()
                n_score = self.rouge.n_score(ref_texts, cand_text)
                n_scores[system_id][topic_id] = n_score

        n_scores_perl = PerlScriptUtils.parse_rouge_perl_out("duc2005_subset/rouge_perl_stem.out")
        for system_id, pyrouge_topics in n_scores.items():
            for topic_id, pyrouge_scores in pyrouge_topics.items():
                scores_perl = n_scores_perl[system_id][topic_id]
                for n, scores in scores_perl.items():
                    for k,v in scores.items():
                        # ROUGE truncates, while we round.
                        self.assertAlmostEqual(v, pyrouge_scores[n][k], 4,
                            "Results different from original ROUGE (stem).")

        # stem, swr
        self.rouge = Rouge.from_rouge155_args({"m": True, "s": True})
        n_scores = {}
        for system_id, cand_texts in tqdm(systems.items()):
            if system_id not in n_scores:
                n_scores[system_id] = {}
            for topic_id, cand_text in cand_texts.items():
                if topic_id not in n_scores[system_id]:
                    n_scores[system_id][topic_id] = {}
                ref_texts = models[topic_id].values()
                n_score = self.rouge.n_score(ref_texts, cand_text)
                n_scores[system_id][topic_id] = n_score

        n_scores_perl = PerlScriptUtils.parse_rouge_perl_out("duc2005_subset/rouge_perl_stem_swr.out")
        for system_id, pyrouge_topics in n_scores.items():
            for topic_id, pyrouge_scores in pyrouge_topics.items():
                scores_perl = n_scores_perl[system_id][topic_id]
                for n, scores in scores_perl.items():
                    for k,v in scores.items():
                        # ROUGE truncates, while we round.
                        self.assertAlmostEqual(v, pyrouge_scores[n][k], 4,
                            "Results different from original ROUGE (stem, swr).")

if __name__ == '__main__':
    unittest.main()