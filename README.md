# Intent-Classification-with-BERT

BERT is an open source model released by Google. I used the pre trained BERT from [Hugging face Pytorch transformers](https://huggingface.co/transformers/). 

## Dataset

The dataset used was ATIS ( Airline Travel Information System) and it a preprocessed version was downloaded from [here](https://www.kaggle.com/siddhadev/atis-dataset-from-ms-cntk.)

This data set contains 4978 train and 893 test spoken utterances (text) classified into one of *26 intents*.

The following are few examples of query and intent,

```
Query text: BOS how much does dl 746 cost EOS
Intent label:  airfare

Query text: BOS what flights from milwaukee to san jose on wednesday on american airlines EOS
Intent label:  flight
```

## Model

The *pretrained BERT base uncased version* was used as the base and the **BERT for sequence classification** model was used where a single linear layer was added on top for classification. 

## Accuracy

The final training loss and validation set accuracy are as follows,

```
Train loss: 0.01674700544348785
Validation set Accuracy: 0.994140625
```

The test set accuracy was 94.95%

With the help of BERT, models with high accuracy can be built using transfer learning.