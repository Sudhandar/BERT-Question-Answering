# Intent-Classification-with-BERT

A multiclass classifcation problem containing text data as input feature instead of multiple feature columns solved using a pretrained deep learning model, BERT from Google.

## Dataset

The dataset used for this task is known as  ATIS ( Airline Travel Information System) and  the pickle files can be downloaded from [here](https://www.kaggle.com/siddhadev/atis-dataset-from-ms-cntk.)

| Entity        		| Value         	|
| ----------------------|:-----------------:|
| No of rows    		| 5871				|
| No of target classes  | 26      			|
| Distribution 			| Skewed     		|
| Cross-validation 		| Stratified K-fold |
| Train-Test split 		| 85% - 15%     	|
| Accuracy Metric 		| Weighted F1 Score |

The following are few examples of query and intent,
```
Query text: how much does dl 746 cost
Intent label:  airfare

Query text: how much does it cost to rent a car in tacoma
Intent label:  ground_fare
```
## Model

*Bert for Sequence Classification* was used for this process. The tranformers library for huggingface provides the pretrained Bert model which was used in this project. The following are the model specifics,

| Entity        		| Value         	|
| ----------------------|:-----------------:|
| Epochs   				| 10				|
| Batch size			| 32      			|
| Optimizer 			| Adam Optimizer    |

## Accuracy

The final F1 socre on the test data is 0.98,

```
F1 Score (weighted): 0.9854597408262985
```
Using transfer learning, high accuracy can be achieved for normal day to day datasets which would otherwise be computationaly expensive and time consuming.


