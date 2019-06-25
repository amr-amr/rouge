# rouge
Pure python implementation of ROUGE, designed for:
 - Faithful reproduction of ROUGE 1.5.5 results;
 - Extendability with customizable tokenization and scoring methods;
 - Incremental scoring for use in e.g. reinforcement learning rewards, 
   or beam search.

## In development:  
- ROUGE-L;
- Ordered ROUGE-N (information ordering aware version of ROUGE-N);
- Test on more data to confirm ROUGE 1.5.5 reproducibility;

## Examples:
```python
from rouge import Rouge
from pprint import PrettyPrinter
pp = PrettyPrinter()

# default args used for unspecified keys
rouge155_args = dict(
    b=0,  # only use first n bytes in system/peer summary for evaluation
    l=0,  # only use first n words in system/peer summary for evaluation
    m=False,  # Porter stemmer
    s=False,  # Stopword removal
    
    n=4,    # Compute up to ROUGE-n
    f="A",  # Scoring formula ('A': model average, 'B': best model)
    p=0.5,  # relative recall/precision importance for F-Score,
    
    # args not implemented
    x=None,  # don't compute ROUGE-L
    c=None,  # confidence interval for bootstrap resampling 
    r=None,  # number of sampling points in bootstrap resampling
    d=None,  # compute per-evaluation average score
    v=None,  # verbose debugging prints
    w=None,  # ROUGE-W weight
    e=None,  # ROUGE_EVAL_HOME directory
    z=None,  # ROUGE-eval-config-file
    t=None,  # Counting unit for averaging
    )

rouge = Rouge.from_rouge155_args(rouge155_args)
```
#### Multiple references
```python
reference_texts = ["The cat was under the bed.",
                   "The sneaky kitty was hiding under the sleeping furniture",
                   ]
candidate_text = "The kitty was sneakily hiding under the bed."
pp.pprint(rouge.n_score(reference_texts, candidate_text))
```
```python
{'ROUGE-1': {'F': 0.70968, 'P': 0.6875, 'R': 0.73333},
 'ROUGE-2': {'F': 0.37037, 'P': 0.35714, 'R': 0.38462},
 'ROUGE-3': {'F': 0.17391, 'P': 0.16667, 'R': 0.18182},
 'ROUGE-4': {'F': 0, 'P': 0.0, 'R': 0.0}}
```


#### Single reference
```python
reference_texts = ["The cat was under the bed."]
candidate_text = "The kitty was sneakily hiding under the bed."
pp.pprint(rouge.n_score(reference_texts, candidate_text))
```
```python
{'ROUGE-1': {'F': 0.71429, 'P': 0.625, 'R': 0.83333},
 'ROUGE-2': {'F': 0.33333, 'P': 0.28571, 'R': 0.4},
 'ROUGE-3': {'F': 0.2, 'P': 0.16667, 'R': 0.25},
 'ROUGE-4': {'F': 0, 'P': 0.0, 'R': 0.0}}
```

#### Incremental scoring

```python
reference_texts = ["The cat was under the bed.",
                   "The sneaky kitty was hiding under the sleeping furniture",
                   ]
candidate_text_gen = "The kitty was sneakily hiding under the bed.".split()
rouge.reset_incremental(reference_texts)
for word in candidate_text_gen:
    print(f"Word: {word}")
    pp.pprint(rouge.n_score_incremental(word))
rouge.reset_incremental(reference_texts)
```

```python
Word: The
{'ROUGE-1': {'R': 0.13333333333333333},
 'ROUGE-2': {'R': 0.0},
 'ROUGE-3': {'R': 0.0},
 'ROUGE-4': {'R': 0.0}}
Word: kitty
{'ROUGE-1': {'R': 0.06666666666666667},
 'ROUGE-2': {'R': 0.0},
 'ROUGE-3': {'R': 0.0},
 'ROUGE-4': {'R': 0.0}}
Word: was
{'ROUGE-1': {'R': 0.13333333333333333},
 'ROUGE-2': {'R': 0.07692307692307693},
 'ROUGE-3': {'R': 0.0},
 'ROUGE-4': {'R': 0.0}}
Word: sneakily
{'ROUGE-1': {'R': 0.0},
 'ROUGE-2': {'R': 0.0},
 'ROUGE-3': {'R': 0.0},
 'ROUGE-4': {'R': 0.0}}
Word: hiding
{'ROUGE-1': {'R': 0.06666666666666667},
 'ROUGE-2': {'R': 0.0},
 'ROUGE-3': {'R': 0.0},
 'ROUGE-4': {'R': 0.0}}
Word: under
{'ROUGE-1': {'R': 0.13333333333333333},
 'ROUGE-2': {'R': 0.07692307692307693},
 'ROUGE-3': {'R': 0.0},
 'ROUGE-4': {'R': 0.0}}
Word: the
{'ROUGE-1': {'R': 0.13333333333333333},
 'ROUGE-2': {'R': 0.15384615384615385},
 'ROUGE-3': {'R': 0.09090909090909091},
 'ROUGE-4': {'R': 0.0}}
Word: bed.
{'ROUGE-1': {'R': 0.06666666666666667},
 'ROUGE-2': {'R': 0.07692307692307693},
 'ROUGE-3': {'R': 0.09090909090909091},
 'ROUGE-4': {'R': 0.0}}
Word: 
{'ROUGE-1': {'R': None},
 'ROUGE-2': {'R': None},
 'ROUGE-3': {'R': None},
 'ROUGE-4': {'R': None}}

```
