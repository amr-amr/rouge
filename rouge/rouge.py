from typing import Dict, List
RougeScores = Dict[str, Dict[str, float]] # {'ROUGE-1': {'R': 0., 'P': 0., 'F': 0.}}
from rouge.utils.tokenize import BaseTokenizer, Rouge155Tokenizer
from collections import Counter
from itertools import chain

# TODO: bootstrap resampling
# TODO: ROUGE-L
# TODO: ROUGE-W
# TODO: ROUGE-SU
# TODO: ROUGE-BE


class Rouge(object):
    def __init__(self, tokenizer: BaseTokenizer, n: int, scoring: str, alpha: float):
        self.tokenizer = tokenizer
        self.N = n
        self.scoring = scoring
        self.alpha = alpha

        self.incremental = {}

    @classmethod
    def from_rouge155_args(cls, args: Dict = None):

        default_args = dict(
            b=0,  # only use first n bytes in system/peer summary for evaluation
            l=0,  # only use first n words in system/peer summary for evaluation
            m=False,  # Porter stemmer
            s=False,  # Stopword removal

            n=4,  # Compute up to ROUGE-n
            f="A",  # Scoring formula ('A': model average, 'B': best model)
            p=0.5,  # relative recall/precision importance for F-Score,

            # args not implemented
            x=False,  # don't compute ROUGE-L # TODO: ROUGE-L
            c=0.95,  # confidence interval for bootstrap resampling # TODO: bootstrapping
            r=1000,  # number of sampling points in bootstrap resampling
            d=True,  # compute per-evaluation average score
            v=False,  # verbose debugging prints
            w=None,  # TODO: rouge-w
            e=None,  # TODO: custom filepath for ROUGE data dirs
            z=None,  # TODO: file integration
            t=None,  # TODO: BE
        )
        if args is not None:
            for k,v in default_args.items():
                if k not in args:
                    args[k] = v
        else:
            args = default_args

        tokenizer = Rouge155Tokenizer(byte_limit=args['b'], word_limit=args['l'], stem=args['m'], sw_removal=args['s'],
                                      sentence_split="SPL")
        return cls(tokenizer, n=args['n'], scoring=args['f'], alpha=args['p'])

    def n_score(self, ref_texts: List[str], cand_text: str) -> Dict[str, Dict[str, float]]:
        """Calculate ROUGE-N scores for a candidate text and reference texts.

        Note:   in the original ROUGE 1.5.5 script, reference texts are referred to as "models"
                while candidate texts are referred to as "peers" or "systems".

        :param ref_texts:   List of reference texts
        :param cand_text:   Candidate text
        :return results:    Nested dictionary of results where first keys are e.g. `ROUGE-1`, `ROUGE-2` and
                            second keys are `R`, `P`, `F`; for recall, precision and f-score respectively.
        """
        results = {}
        cand_text_tokens = self._tokenize(cand_text)
        ref_texts_tokens = [self._tokenize(ref) for ref in ref_texts]
        for n in range(1, self.N+1):
            cand_counter = self.generate_counter(self._ngram_tokenize(cand_text_tokens, n))
            cand_count = sum(cand_counter.values())
            refs_count = []
            cands_count = []
            matches_count = []
            for ref in ref_texts_tokens:
                ref_counter = self.generate_counter(self._ngram_tokenize(ref, n))
                refs_count += [sum(ref_counter.values())]
                cands_count += [cand_count]
                matches = {ngram: min(ref_counter[ngram], cand_counter[ngram]) for ngram in cand_counter.keys()
                           if ngram in ref_counter}
                matches_count += [sum(matches.values())]

            if self.scoring == "A":
                cands_count, refs_count, matches_count = sum(cands_count), sum(refs_count), sum(matches_count)

            elif self.scoring == "B":
                cands_count, refs_count, matches_count = max(zip(cands_count, refs_count, matches_count),
                                                             key=lambda x: x[2]/x[1])

            recall = matches_count/refs_count if refs_count else 0
            precision = matches_count/cands_count if cands_count else 0
            fscore = (precision*recall) / ((1-self.alpha)*precision + self.alpha*recall) if matches_count else 0

            
            results[f"ROUGE-{n}"] = {"R": round(recall, 5), "P": round(precision, 5), "F": round(fscore, 5)}

        return results

    def n_score_incremental(self, new_text: str):
        """Calculates incremental increase in ROUGE-N (recall) given new text (e.g. sentence or word).

        The sum of all incremental scores should be equal to the overall recall score (using "average" scoring method).
        This does not apply to precision or f-score, so they are not calculated.
        Similarly, this does not apply to the "best" scoring method, so it is not calculated.

        :param new_text:
        :return:
        """
        results = {}
        new_tokens = list(chain.from_iterable(self._tokenize(new_text)))
        if new_tokens:
            self.incremental['prev_tokens'] += new_tokens
            for n in range(1,self.N+1):
                matches_count = 0
                new_ngrams =  self._ngram_tokenize([self.incremental['prev_tokens'][-n:]], n)
                for ref_counter in self.incremental['ref_counters'][n]:
                    for ngram in new_ngrams:
                        if ngram in ref_counter:
                            matches_count += 1

                recall = matches_count/self.incremental['refs_count'][n]

                results[f"ROUGE-{n}"] = {"R": recall}
        else:
            results = {f"ROUGE-{n}": {"R": None} for n in range(1, self.N+1)}
        return results

    def reset_incremental(self, ref_texts: List[str]):
        self.incremental['ref_counters'], self.incremental['refs_count'] = {}, {}
        self.incremental['prev_tokens'] = ["" for _ in range(self.N)] # init with empty tokens to prevent index error

        for n in range(1, self.N+1):
            ref_counters = [self.generate_counter(self._ngram_tokenize(self._tokenize(ref), n)) for ref in ref_texts]
            self.incremental['ref_counters'][n] = ref_counters
            self.incremental['refs_count'][n] = sum(sum(ref_counter.values()) for ref_counter in ref_counters)

    def generate_counter(self, ngrams: List[str]) -> Dict[str, int]:
        """Generate dictionary with ngram counts for a given summary.

        Note:   this one-liner method is included because of plans to extend it beyond a simple counter;
                for example, including ngram positions in a text to allow evaluating information ordering.

        :param ngrams: List of ngrams for a given text
        :return count_dict: Dictionary of ngram counts where `counter[ngram] = ngram_count`
        """
        return Counter(ngrams)

    def _ngram_tokenize(self, tokenized_sentences: List[List[str]], n: int) -> List[str]:
        """Generate list of ngrams from tokenized text.

        :param tokenized_text: List of sentences where each sentence is a list of tokens (1grams)
        :param n: Ngram size
        :return ngrams:  List of ngrams
        """
        tokenized_text = list(chain.from_iterable(tokenized_sentences))
        return [" ".join(tokenized_text[i:i+n]) for i in range(len(tokenized_text)-n+1)]

    def _tokenize(self, text: str) -> List[List[str]]:
        """Preprocess and tokenize text into sentences and 1grams.

        :param text: text to be tokenized
        :return: List of sentences where each sentence is a list of 1gram tokens
        """
        return self.tokenizer.tokenize_text(text)






#
# class Rouge155(Rouge):
#     # TODO: resampling + confidence
#     # TODO: ngrams attribute
#     """
#     "Options:\n".
#       "  -2: Compute skip bigram (ROGUE-S) co-occurrence, also specify the maximum gap length between two words (skip-bigram)\n".
#       "  -u: Compute skip bigram as -2 but include unigram, i.e. treat unigram as \"start-sentence-symbol unigram\"; -2 has to be specified.\n".
#       "  -3: Compute BE score. Currently only SIMPLE BE triple format is supported.\n".
#       "      H    -> head only scoring (does not applied to Minipar-based BEs).\n".
#       "      HM   -> head and modifier pair scoring.\n".
#       "      HMR  -> head, modifier and relation triple scoring.\n".
#       "      HM1  -> H and HM scoring (same as HM for Minipar-based BEs).\n".
#       "      HMR1 -> HM and HMR scoring (same as HMR for Minipar-based BEs).\n".
#       "      HMR2 -> H, HM and HMR scoring (same as HMR for Minipar-based BEs).\n".
#       "  -a: Evaluate all systems specified in the ROUGE-eval-config-file.\n".
#       "  -c: Specify CF\% (0 <= CF <= 100) confidence interval to compute. The default is 95\% (i.e. CF=95).\n".
#                 default: 95% confidence interval;
#                 # TODO: implement this
#       "  -d: Print per evaluation average score for each system.\n".
#       "  -e: Specify ROUGE_EVAL_HOME directory where the ROUGE data files can be found.\n".
#       "      This will overwrite the ROUGE_EVAL_HOME specified in the environment variable.\n".
#       "  -f: Select scoring formula: 'A' => model average; 'B' => best model\n".
#                 default: A
#       "  -h: Print usage information.\n".
#       "  -H: Print detailed usage information.\n".
#       "  -b: Only use the first n bytes in the system/peer summary for the evaluation.\n".
#                 default: 0
#                 # TODO: implement this
#       "  -l: Only use the first n words in the system/peer summary for the evaluation.\n".
#                   default: 0
#                 # TODO: implement this
#       "  -m: Stem both model and system summaries using Porter stemmer before computing various statistics.\n".
#                 default: no stemming
#       "  -n: Compute ROUGE-N up to max-ngram length will be computed.\n".
#                 # TODO: implement this
#       "  -p: Relative importance of recall and precision ROUGE scores. Alpha -> 1 favors precision, Alpha -> 0 favors recall.\n".
#                 default: 0.5
#                 # TODO: implement this
#       "  -s: Remove stopwords in model and system summaries before computing various statistics.\n".
#                 default: stopwords not removed
#       "  -t: Compute average ROUGE by averaging over the whole test corpus instead of sentences (units).\n".
#       "      0: use sentence as counting unit, 1: use token as couting unit, 2: same as 1 but output raw counts\n".
#       "      instead of precision, recall, and f-measure scores. 2 is useful when computation of the final,\n".
#       "      precision, recall, and f-measure scores will be conducted later.\n".
#                 default: 0
#                 # TODO: figure out wtf this means, then implement
#       "  -r: Specify the number of sampling point in bootstrap resampling (default is 1000).\n".
#       "      Smaller number will speed up the evaluation but less reliable confidence interval.\n".
#                 default: 1000
#                 # TODO: figure out wtf this means, then implement
#       "  -w: Compute ROUGE-W that gives consecutive matches of length L in an LCS a weight of 'L^weight' instead of just 'L' as in LCS.\n".
#       "      Typically this is set to 1.2 or other number greater than 1.\n".
#                 # TODO: implement this
#       "  -v: Print debugging information for diagnositic purpose.\n".
#       "  -x: Do not calculate ROUGE-L.\n".
#                 default: calculates ROUGE-L
#                 # TODO: implement this
#       "  -z: ROUGE-eval-config-file is a list of peer-model pair per line in the specified format (SEE|SPL|ISI|SIMPLE).\n";
#
#     """
