# BERT Questing Answering model on SQUAD dataset

Link to view the notebook: [Github view](https://github.com/Sudhandar/BERT-Question-Answering/blob/master/BERT_Question_and_Answering.ipynb)

Link to the Colab Notebook: [Colab Notebook](https://colab.research.google.com/drive/1uYtqckEAp6DwweZGw4VxyU9rvEax-A5y?authuser=1#scrollTo=WqR9cK3zblzA)

Fine tuning the BERT base-cased model to build a question and answering model,trained and tested on the SQuAD dataset.

## About BERT

BERT( Bidirectional Encoder Representations from Transforers) method of pre-training language representations. With the use of pre-trained BERT models we can utilize a pre-trained memory information of sentence structure, language and text grammar related memory of large corpus of millions, or billions, of annotated training examples that it has trained. BERT makes use of Transformer, an attention mechanism that learns contextual relations between words (or sub-words) in a text.The pre-trained model can then be fine-tuned on small-data NLP tasks like question answering and sentiment analysis, resulting in substantial improvements in accuracy compared to training on these datasets from scratch. The following is the structure of BERT,

![alt text](https://github.com/Sudhandar/BERT-Question-Answering/blob/master/images/bert_structure.png)

[Image Source](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/blocks/bert-encoder)

## SQUAD Dataset

Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
SQuAD 1.1 contains 100,000+ question-answer pairs on 500+ articles. The data is stored in the json format. After preprocessing, the following is the structure followed in the code,

* **qas_id**: The example's unique identifier
* **title**: Article title
* **question_text**: The question string
* **context_text**: The context string
* **answer_text**: The answer string
```
Title: University_of_Notre_Dame
ID: 5733caf74776f4190066124c

======== Question =========
How many wins does the Notre Dame men's basketball team have?

======== Context =========
The men's basketball team has over 1,600 wins, one of only 12 schools who have
reached that mark, and have appeared in 28 NCAA tournaments. Former player
Austin Carr holds the record for most points scored in a single game of the
tournament with 61. Although the team has never won the NCAA Tournament, they
were named by the Helms Athletic Foundation as national champions twice. The
team has orchestrated a number of upsets of number one ranked teams, the most
notable of which was ending UCLA's record 88-game winning streak in 1974. The
team has beaten an additional eight number-one teams, and those nine wins rank
second, to UCLA's 10, all-time in wins against the top team. The team plays in
newly renovated Purcell Pavilion (within the Edmund P. Joyce Center), which
reopened for the beginning of the 2009–2010 season. The team is coached by Mike
Brey, who, as of the 2014–15 season, his fifteenth at Notre Dame, has achieved a
332-165 record. In 2009 they were invited to the NIT, where they advanced to the
semifinals but were beaten by Penn State who went on and beat Baylor in the
championship. The 2010–11 team concluded its regular season ranked number seven
in the country, with a record of 25–5, Brey's fifth straight 20-win season, and
a second-place finish in the Big East. During the 2014-15 season, the team went
32-6 and won the ACC conference tournament, later advancing to the Elite 8,
where the Fighting Irish lost on a missed buzzer-beater against then undefeated
Kentucky. Led by NBA draft picks Jerian Grant and Pat Connaughton, the Fighting
Irish beat the eventual national champion Duke Blue Devils twice during the
season. The 32 wins were the most by the Fighting Irish team since 1908-09.

======== Answer =========
over 1,600
```

## Labelling the Answers

**Replacing Answer with [MASK] Tokens**

The strategy is to modify the context string and replace the answer with a "sentinel string"--a unique and easily identifiable sequence of tokens--before feeding the context string through the tokenizer. 

* **Step 1**: Feed the **answer** string into the BERT tokenizer to determine the number of tokens it breaks into.
* **Step 2**: In the context, replace the answer with a string made up of `[MASK]` tokens, matching the number of tokens in the answer. 
* **Step 3**: Feed the modified text into the BERT tokenizer to tokenize and encode everything.
* **Step 4**: Locate the `[MASK]` tokens in the encoded result. This is easy, since the `[MASK]` token has a specific ID and won't appear anywhere else in the sequence.
* **Step 5**: Record the start and end indices of the answer, and finally "repair" the encoded sequence by switching the MASK token IDs out for the original answer token IDs.


## Sequence Length Distribution

Part of tokenizing and encoding text for BERT is choosing a **maximum sequence length (max_len)** to pad or truncate all of the samples in the training set (87,599 samples). The BERT tokenizer.encode does the following steps,

1. Split the sentence into tokens.
2. Add the special `[CLS]` and `[SEP]` tokens.
3. Map the tokens to their IDs.

Here are the minimum, maximum, and median sequence lengths.

```
Min length: 36 tokens
Max length: 882 tokens
Median length: 163 tokens
```

The distribution plot of the sequence lengths, 

![alt text](https://github.com/Sudhandar/BERT-Question-Answering/blob/master/images/sequence_length_distribution.png)


Finally, the number of training samples which would be impacted, given a handful of different choices of `max_len`.

```
Number of comments truncated based on max_len,

max_len = 128  -->   69,082 of  87,599  (78.9%)  will be truncated 
max_len = 256  -->    9,812 of  87,599  (11.2%)  will be truncated 
max_len = 300  -->    4,568 of  87,599  ( 5.2%)  will be truncated 
max_len = 384  -->    1,087 of  87,599  ( 1.2%)  will be truncated 
max_len = 512  -->      135 of  87,599  ( 0.2%)  will be truncated 
```
There are several factors that impact our choice of the maximum sequence length `max_len`:

1. **Training Time** - Training time is quadratic with `max_len`. `max_len = 512` will take 4x  longer to train than `max_len = 256`, and 16x longer than `max_len = 128`!
2. **Accuracy** - Truncating the samples to a shorter length will presumably hurt accuracy, due to the loss of information. 
3. **GPU Memory** - The combination of `max_len` and `batch_size` need to fit within the memory limits of Google Colab's GPU. For a Tesla K80 (which has 12GB of RAM), with `batch_size = 16`, the maximum length that can be used (without running of memory) is about `max_len = 400`.

A maximum sequence length of `384` is used as suggested by the BERT authors.

## Hyperparameters Used

The model object handles the execution of a forward pass, and the calculation of gradients during training.The actual updates to the model's weights, however, are performed by an Optimizer object.It is given as  a reference to the model's parameters, as well as set some of the training hyperparameters.

For the purposes of fine-tuning, the BERT authors recommend choosing from the following values:

* Learning rate (Adam): 5e-5, 3e-5, 2e-5
* Number of epochs: 2, 3, 4 ("learning rate scheduler")

In order to make more efficient use of the GPU's parallel processing capabilities, a batch size of 12 is used. The learning rate used is 3e-5 and number of epochs is 2.

## Training and Validation

This dataset already has a train / test split, but the training dataset has been further divided to use 98% for training and 2% for *validation*. The validation set is used to detect over-fitting during the training process.

There are 7,105 training batches & 145 validation batches for 1 epoch.The dataset was trained for 3.5 hours on GPU.The follwowing table shows the loss in each epochs,

| Epoch | Training Loss | Validation Loss |	Validation Accuracy | Training Time	| Validation Time |				
|-------|---------------|-----------------|---------------------|---------------|-----------------|
|  1	|    1.23	    |      0.92	      |         0.72	    |     1:47:19	|      0:00:45    |
|  2	|    0.70       |	   0.93	      |         0.73	    |     1:47:18	|      0:00:45    |

## Evaluation on test set

The SQuAD test set follows the same json structure as the training set, however, there are 3 answers provided for every question. These are three human-provided answers, and they don't always agree. For example, for the question:

```
Where did Super Bowl 50 take place?
```

The annotators produced:
```
   {'answer_start': 403, 'text': 'Santa Clara, California'}
   {'answer_start': 355, 'text': "Levi's Stadium"}
   {'answer_start': 355, 'text': "Levi's Stadium in the San Francisco Bay Area at Santa Clara, California."}
```   

Since all three of these seem acceptable,BERT's prediction to all three correct answers are compared, and the highest F1 score that BERT gets among the three is considered.

For the test samples, a 2-pass approach has been followed to tokenize the samples. 

In the first pass,all the samples are tokenized **without any truncation or padding**, which allows us to correctly locate the answers, even if their token indices are greater than 384.

In the second pass, the samples are tokenized and encoded , with padding and truncation.

The test set contains 21140 samples.

## Final Results

There are two standard approaches to scoring results on the SQuAD benchmark:

1. Exact Match
2. F1 Score

**Exact Match**

For this metric, the number of predicted start indices that are equal to the correct ones are added up. It is done for  the end indices as well, such that there are actually two "points" for every sample.

To handle the 3 possible answers, we score our predictions against each of the answers separately, and select the  answer which best matches our prediction. So for each test sample, the highest possible score is 2. 

**F1 Score**

The F1 score gives our model credit for predicting a span which partially intersects the correct one.

*Formula :* 
```python
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
```

To handle the 3 possible answers, F1 score for each sample is calculated separately and the one with the highest score is considered. 

The final F1 score is determined by taking the average over all the test samples.

**Final Score of our fine tuned BERT base model:**

```
Correctly predicted indices: 17,751 of 21,140 (83.97%)

Average F1 Score: 0.863
```

## References

1. https://datarepository.wolframcloud.com/resources/SQuAD-v1.1
2. https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/blocks/bert-encoder
3. https://rajpurkar.github.io/SQuAD-explorer/
4. https://github.com/google-research/bert