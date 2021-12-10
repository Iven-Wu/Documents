Universal Sentence Encoder
Defense:

* Empirical Defense
	* Adversarial Data Augmentation: [MixADA](https://arxiv.org/pdf/2012.15699.pdf)
	* Adversarial Training: [PGD-K](https://openreview.net/pdf?id=rJzIBfZAb), [FreeLB](https://openreview.net/pdf?id=BygzbyHFvB), [TAVAT](https://arxiv.org/pdf/2004.14543.pdf), [InfoBERT](https://arxiv.org/pdf/2010.02329.pdf)
	* Region-based adversarial Training: [DNE](https://arxiv.org/pdf/2006.11627.pdf), [ASCC](https://arxiv.org/pdf/1812.05271.pdf)
* Certified Defense
	* Interval Bound Propagation: [LSTM-based](https://aclanthology.org/D19-1423.pdf), [Transformers](https://arxiv.org/pdf/2002.06622.pdf)
	* Randomized Smoothing: [SAFER](https://aclanthology.org/2020.acl-main.317.pdf), [RanMASK](https://arxiv.org/pdf/2105.03743.pdf)

Attack:

* [PWWS](https://aclanthology.org/P19-1103.pdf)
* [TextAttack](https://arxiv.org/pdf/2005.05909.pdf)
* [TextBugger](https://arxiv.org/pdf/1812.05271.pdf)
* [TextFooler](https://arxiv.org/pdf/1907.11932.pdf)
* [BERT-Attack](https://aclanthology.org/2020.emnlp-main.500.pdf)

---

### Benchmarking Defense against Adversarial Word Substitution

#### Abstract

Recent studies show that DNN are vulnerable to intentionally crafted adversarial examples, lots of methods have been proposed to defend against word-substitution attack for NLP. However, there is a lack of systematic study on comparing defense approawches under the same attcking setting, which is so called **benchmark**.

Dataset: AGNEWS and IMDB.

#### Intorduction

**Goal: **

The goal of adversarial defenses is to learn a model that is capable of achieving high test accuracy on both clean and adversarial examples.

**Problem:**

* No standard comparison for defense method.
* Except for RanMASK, all existing defense methods assume that the defenders are informed of how the adversaries generate synonyms.

**Solutions:**

* clean accuracy
* accuracy under attack
* attack success rate
* number of queries

Improvement of FreeLB

#### Background

**Textual Adversarial Attacks**

 The textual adversarial attacks can be formulated as the following equation:
$$
f(\pmb{x'}) \neq \pmb{y}, \quad Sim(\pmb{x}, \pmb{x'}) \geq \epsilon_{min}
$$
**Adversarial Word Substitution**

* Determine an important position to change
* Modify words in the selected positions to maximize the model's prediction error.
  * Greedy algorithm
  * Combinatorial optimization algorithms

#### Constraints on Adversarial Example

**Constraints on adversaries**

* The minimum semantic similarity $\epsilon_{min}$ between original input $\pmb{x}$ and adversarial example $\pmb{x'}$
* The maximum number of one word's synonyms $K_{max}$
* The maximum percentage of modified words $\rho_{max}$
* The maximum number of queries to the victim model $Q_{max}$

*Semantic Similarity:* Using the Universal Sentence Encoder to evaluate the semantic similarity, which encodes the sentence  the sentence into a vectore and use cosine similarity score to identify the similarity.
*Percentage of Modified  Words:* An attacker is not allowed to perturb too many words since the more wwords in a sentence are perturbed, the lower the similarity with the original sentence. 

*Number of Queries*

**Dataset and Hyper-parameter**

5 representative adversarial word substitution algorithms:

* PWWS
* DeepWordBug
* TextBugger
* TextFooler
* BERT-Attack

---

#### BERT-ATTACK: 

**Methods**

1. Finding Vulerable Words
2. Word Replacement via BERT

----

#### PWWS:

**Method**

1. Word Substitution Strategy

   For each word $w_i \in \mathbf{x}$, we use WordNet to build a synonym set $\mathcal{L}_i$. Find the $w^*_i \in \mathcal{L}$ such that make the biggest different to the output.

2. Replacement Order Strategy

   1. Word saliency refers to the degree of change in theoutput probability of the classifier if a word is set to unknown. The saliency of a word is computed as the difference between output probability.
   2. $H(\mathbf{x}, \mathbf{x}^*_i,w_i) = \phi(S(x))_i * \Delta P^*_i$

---

#### TextBugger:

**Method**

Black-box Attack: Under the black box setting, gradients of the model are not directly available, and we need to change the input sequences directly without the guidance of gradients. Briefly, the proc ess of generating word-based adversarial examples on text under black-box setting contains three steps:

1. Find the important setences

   1. Use the spaCy library to segment each document into sentences. 
   2. Filter out the $\mathcal{F}_l(\mathbf{s_i})\neq y$
   3. Sorted the sentences with their important score 

2. Use a scoring function to determine the importance of each word regarding to the classification.

   1. We intorduce a scoring dunction that determine the importance of the $j^{th}$ word in $x$ as:
      $$
      C_{w_j} = \mathcal{F}_y(w_1,w_2,...,w_m) - \mathcal{F}_y(w_1,...,w_{j-1},w_{j+1},...,w_m)
      $$

3. Use the bug selection algorithm to xhange the selected words

   1. BugGenerator, finding the k-nearest neighbor words to generate bug set, which keep the semantic meaning.
   2. For each word in bug set, replace the bug to the original sentence and pick the $bug_{best}$ which has largest score.

---

#### TextFooler

**Method**

1. Word Importance Ranking:

   We determine the importace of word of a specific sentence by defining the important score for word, by following the TextBugger. However, we consider different scenario for the output of $F(X') \neq y$. (TextBugger filter out the $F(X')\neq y$)

2. Word Transformer:

   1. Synonym Extraction:

      We use word embeddings to find synonyms and further test on SimLex-999

   2. POS Checking:

      POS checking, which can ensure the grammar.

   3. Semantic Similarity Checking:

      We use Universal Sentence Encoder to check the semantic similiarity between adversarial examples and original one.

---

#### TextAttack

**Abstract**

This paper introduces *TextAttack*, a Python framework for adversarial attacks, data augmentation, and adversarial trainning in NLP. It builds attack from four components:

1. A goal functionn
2. A set of constraints
3. A transformation
4. A search method

**Introduction**

The attack attemtps to perturb an input text such that the model output fulfills the fgoal function and the perturbation adheres to the set of constraints. A search method is used to find a sequennce of transformations that produce a successful adversarial example.

TextAttack is directly integrated with HuggingFace's transformers and nlp libraries. 

**Framework**

In such way, attacking an NLP model can be framed as a combinatorial search problem. The attacker must search withn all potential transformations that generate a successful adversarial example.

1. Goal function: It determines whether the attack is successful in terms of the model outputs.
2. A set of constraints: It determine if a perturbation is valid with respect to the original input.
3. Transformation: Given an input, generates a set of potential perturbations.
4. A search method: It successively queries the model and selects promising perturbations form a set of transformations.

**Understand the api and know how to use **

---

#### Defending Against Neural Fake News

**Abstract**

Recent progress in NLP generation has raised dual-use concerns. While applications like summarization and translation are positive, the underlying technology also might enable adversaries to generate neural fake news.

We thus present a model for controllable text generation called **Grover**.

**Fake News in a Neural and Adversarial Setting**

We present a framework motivated by today's dynamic of manually created fake news for understanding what adversaries will attempt with deep models and how verifiers shoud respond.

*Scope of fake news*

Existing fake news is predominantly human-written, for two broad goals:

1. Monetization
2. Propaganda

*Fact checking and verification*

Recently, there are lots of services rely on manual fact-checking efforts. These efforts can help moderators on social media platforms shut down suspicious accounts. However, fact checking is not a panacea-coginitive biases such as the backfire effect and confirmation bias make humans liable to believe fake news that fits their worldview.

*Framework*

We cast fake news generation and detection as an adversarial game, with two players:

1. Adversary: Their goal is to generate fake stories that match specified attributes: generally, being viral or persuasive. The stories must read realistically to both human users as well as the verifier.
2. Verifier: Their goal is to classify news stories as real or fake. Theverifier has access to unlimited real news stories, but few fake news stories from a specific adversary.

The dual objective of these two players suggest an escalating arms race between attackers and defenders. As ver ification systems get better, so too will adversaries. 

**Grover: Modeliong Conditional Generat ion of Neural Fake News**

An article can be modeled by the joint distribution:
$$
p(domain, date, authors,headline,body)
$$
It is difficult to sample form the above equation. In a naive way, the model need to learn to handle $|\mathcal{F}|!$ potential orderings during infernece time.

Our solution is Grover, a new approach for efficient learning and generation of multi-field documents. During inference time, we start with a set of fields $\mathcal{F}$ as context, with each field $f$ containing field specific start and end tokens. We sorth the fields using a standard order and combine the resulting tokens together.

Traing, it used two set but why? 

**Neural Fake News Detection**

The high quality of neural fake news written by Grover, as judged by humans, make automatic neural fake news detection an important research area. Using model for the role of the verifier can mitigate the harm of nerual fake news by classifying articles as Human or Machine written. These decisions can assist content moderators and end users in identifying neural disinformation.

---

#### FreeLB: ENHANCED ADVERSARIAL TRAINING FOR NATURAL LANGUAGE UNDERSTANDING

**Abstract**

Adversarial training, which minimizes the maximal risk for label-perserving input perturbations, has proved to be effective for imporving the generalization of language model. We proposed FreeLB, that promotes higher invariance in the embedding space. To validiate the effectiveness of the proposed approach, we apply it to transformer-based models for natural language understanding and commonsense reasoning tasks.

**Introduction**

In particular, we propose a novel adversarial trianing algorithm, called FreeLB, which adds adversarial perturbations to word embeddings and minimizes the resultant adversarial loss aroung input samples. The method leverages recenntly proposed 'free' 











