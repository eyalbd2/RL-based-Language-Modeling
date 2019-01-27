# Improving Language Models Using Reinforcement Learning

In this work, we show how to use traditional
evaluation methods used in NLP to improve a language model. We construct
couple of models, using different objective for each model, and comparing between
them, while offering some intuition regarding the results.

The project implementation contains:
1.	Basic Cross-entropy seq2seq model training - Using couple of tricks to get a faster training convergence
2.	4 Policy Gradient model - using 4 different rewards (for further explanation, please refer to "Deep Reinforcement Learning For Language Models.pdf", which is in this repository): 
	- [ ] BLUEU score
	- [ ] Maximum Mutual Information score
	- [ ] Perplexity score
	- [ ] Cosine Similarity score
3.	Testing generated dialogues - we test each of our models and sum up the results (for all results, please refer to "Deep Reinforcement Learning For Language Models.pdf", which is in this repository).  
4.	Automatic evaluation - we evaluate each of the models using traditional NLP evaluation methods.


<p align="center">
  <img src="https://raw.githubusercontent.com/eyalbd2/LanguageModel-UsingRL/master/Images/many generated sequences.PNG" width="400" title="Many Generated Sequences">
</p>
Here we present a comparison of target sentences generation given a source sequence between Perplexity, MMI, BLEU and Seq2Seq.
<br/>
<br/>
<br/>
<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/eyalbd2/LanguageModel-UsingRL/master/Images/MMI comparison.PNG" width="400" title="MMI comparison">
</p>
This is a comparison of target sentences generation given a source sequence between a model of MMI trained using reinforcement learning policy gradient and a model which is calculating MMI only at test time.
 

## Getting Started

- [ ] Install and arrange an environment that meets the prerequisites (use below subsection)
- [ ] Pycharm is recommended for code development
- [ ] Clone this directory to your computer
- [ ] Call main using suitable arguments from a console for demonstrations or just using code

### Prerequisites

Before starting, make sure you already have the followings, if you don't - you should install before trying to use this code:
- [ ] the code was tested only on python 3.6 
- [ ] numpy
- [ ] scipy
- [ ] pandas
- [ ] pickle


## Running the Code - train, tests, competition
Parameters for main function ( can be seen when writing --help ): 

| Name Of Param     | Description                              | Default Value | Type       | 
| --- | --- | --- | --- | 
| result_dir_name   | path to results directory                |               |            | 
| data_set          | which data-set to work on - 1 or 2       | 1             | int        | 
| operatin_mode     | Choose between train\test\comp           | 'comp'        | str        | 
| reg_lambda        | Constant - coeeficient for regulrization | 3e-05         | np.float64 | 
| threshold         | Constant - feature apearance threshold   | 2             |  int       | 


This project solves works on two differen data-saets. To test each set you need to run different commands.

### First Set
Run Training:
```
python main.py <Result Directory Name> -data_set <1 (also default)> -operation_mode <'train'> -reg_lambda <Pick a float number> -threshold <Pick an integer>
```

Check performance on test set:
```
python main.py <Result Directory Name> -data_set <1 (also default)> -operation_mode <'test'> -reg_lambda <*Keep Empty*> -threshold <*Keep Empty*>
```
Build competition atags:
```
python main.py <Result Directory Name> -data_set <1 (also default)> -operation_mode <'comp'(also default)> -reg_lambda <*Keep Empty*> -threshold <*Keep Empty*>
```

### Second Set
Run Training (There is no test set):
```
python main.py <Result Directory Name> -data_set <2> -operation_mode <'train'> -reg_lambda <Pick a float number> -threshold <Pick an integer>
```

Build competition tags:
```
python main.py <Result Directory Name> -data_set <2> -operation_mode <'comp'(also default)> -reg_lambda <*Keep Empty*> -threshold <*Keep Empty*>
```


## Results for tests
WE present below a table with test results on both data-sets. For more results and conclusions, please review 'NLP_Wet_1.pdf' file in this directory.

<p align="center">
  <img src="https://raw.githubusercontent.com/eyalbd2/NLP_Basic/master/Images/result_first_data_set.PNG" width="500" title="Result for first data-set">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/eyalbd2/NLP_Basic/master/Images/results_second_data-set.PNG" width="500" title="Result for second data-set">
</p>

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.



## Authors

* **Eyal Ben David** 


## License

This project is licensed under the MIT License 

## Acknowledgments

* Basic feature - "A Maximum Entropy Model for Part-Of-Speech Tagging", (Ratnaparkhi, 1996)  
* Developed features- "Enriching the Knowledge Sources Used in a Maximum Entropy Part-of-Speech Tagger", (Toutanova and Manning, 2000)