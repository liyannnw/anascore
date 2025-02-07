# AnaScore

This is the code repository for the paper "AnaScore: Understanding Semantic Parallelism in Proportional Analogies".


## Abstract
>  Formulaic criteria for proportional analogies, which capture relational mappings between two ratios of terms, are mainly confined to the formal level. As analogy datasets grow more complex, especially in evaluating the cognitive abilities of Large Language Models (LLMs), assessing parallelism in them becomes increasingly challenging and often requires human annotation. In this work, we propose AnaScore, an automatic metric for evaluating the strength of semantic parallelism in sentence analogies. AnaScore systematically provides formalized explanations for shared relational patterns at the level of conceptual knowledge. We apply AnaScore to annotate several existing datasets, considering different directions of the relations, and uncover artifacts in data construction. Our experiments with various LLMs demonstrate the efficacy of the AnaScore metric in capturing the inherent quality of analogical relationships, showing a positive correlation between analogy quality and model performance. Thanks to this metric, we clearly demonstrate that formally explainable examples are more beneficial for analogical reasoning, while ambiguous analogies with no clear criterion tend to hinder inference.

**TL;DR:** AnaScore computes the degree of parallelism between term descriptions in ConceptNet, checking whether the conceptual transformations from C to D follow the same pattern as those from A to B.



## Python environment
Create a Python virtual environment and install the required dependencies:

```bash
python -m venv venv-anascore
source venv-anascore/bin/activate

pip install -r requirements.txt
```


## Usage

### Compute AnaScore for a given analogy

You can use AnaScore to compute the parallelism score of a given analogy based on term representations in ConceptNet:



```python
from anascore import Anascore

anascore = AnaScore()

quad=["king", "queen", "man","woman"]

outputs= anascore.get_score_from_quad(quad)
print(outputs.score)
```

### Preloading ConceptNet descriptions
To speed up computations, you can pre-fetch conceptual descriptions for all words in the analogies:






```bash
cat ${analogy_fpath} | python3 conceptnet.py --concept_dict ${concept_dict_fpath}
```

* `analogy_fpath`: Path to the file containing analogy instances.
Each line should contain four tab-separated terms representing an analogy in the format.

* `concept_dict_fpath`: Path to save the knowledge graph for words in ConceptNet.




Then, load the ConceptNet dictionary before computing AnaScore:

```python
from utils import load_conceptnet_dict

concept_dict = load_conceptnet_dict(open(concept_dict_fpath))

anascore = AnaScore(concept_dict=concept_dict)
```

### Output details

Output of Anascore includes multiple measures:
* `outputs.quad_non_overlap`: the terms after removing overlapping content
* `outputs.quad_concept`: conceptual representations of four terms using ConceptNet
* `outputs.left_ratio_same`: common concepts in A:B
* `outputs.right_ratio_same`: common concepts in C:D
* `outputs.left_ratio_diff`: different concepts in A:B
* `outputs.right_ratio_diff`: different concepts in C:D
* `outputs.relational_matches`: parallel relational changes in both ratios
* `outputs.rel2parallelism`: degree of parallelism per relational edge
* `outputs.parallelism`: overall parallelism score
* `outputs.content_coverage`: proportion of content that contributes to parallel transformations
* `outputs.score`: final score: a combination of parallelism and content coverage

## Annotation

`annotation.py` supports to classify analogies based on formal and semantic criteria.

Each analogy is labeled with the following categories:



| **Label**         | **Description** |
|-------------------|----------------|
| `formal-char`    | The analogy follows **character-level** transformations (e.g., letter replacements). |
| `formal-word`    | The analogy follows **word-level** transformations |
| `semantic-{X}`   | The analogy does not meet formal criteria but achieves a **semantic similarity score `X`** using AnaScore. |
| `no_criterion`   | The analogy lacks both **formal and semantic justification**. |


The annotated data is available on Huggingface ([liyannn/sentence_analogy](https://huggingface.co/datasets/liyannn/sentence_analogy)).