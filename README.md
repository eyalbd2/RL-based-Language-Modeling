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
  <img src="https://raw.githubusercontent.com/eyalbd2/LanguageModel-UsingRL/master/Images/many generated sequences.PNG" width="1000" title="Many Generated Sequences">
</p>
Here we present a comparison of target sentences generation given a source sequence between Perplexity, MMI, BLEU and Seq2Seq.
<br/>
<br/>
<br/>
<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/eyalbd2/LanguageModel-UsingRL/master/Images/MMI comparison.PNG" width="800" title="MMI comparison">
</p>
This is a comparison of target sentences generation given a source sequence between a model of MMI trained using reinforcement learning policy gradient and a model which is calculating MMI only at test time.
 

## Getting Started

- [ ] Install and arrange an environment that meets the prerequisites (use below subsection)
- [ ] Pycharm is recommended for code development
- [ ] Clone this directory to your computer
- [ ] Train a cross-entropy seq2seq model using 'train_crossent' function
- [ ] Train RL models using appropriate functions (e.g. use 'train_rl_MMI' to use MMI criterion for policy gradient)
- [ ] Test models using automatic evaluations using 'test_all_modells' 
- [ ] Generate responces using 'use_model'

### Prerequisites

Before starting, make sure you already have the followings, if you don't - you should install before trying to use this code:
- [ ] the code was tested only on python 3.6 
- [ ] numpy
- [ ] scipy
- [ ] pickle
- [ ] pytorch
- [ ] TensorboardX


## Running the Code 

### Training all models

#### Train Cross-entropy Seq2Seq and backward Seq2Seq 
Parameters for calling 'train_crossent' ( can be seen when writing --help ): 

| Name Of Param  | Description                             | Default Value | Type       | 
| --- | --- | --- | --- | 
| SAVES_DIR      | Save directory                          | 'saves'       | str        | 
| name           | Specific model saves directory          | 'seq2seq'     | str        | 
| BATCH_SIZE     | Batch Size for training                 | 32            | int        | 
| LEARNING_RATE  | Learning Rate                           | 1e-3          | float      | 
| MAX_EPOCHES    | Number of training iterations           | 100           | int        | 
| TEACHER_PROB   | Probability to force reference inputs   | 0.5           | float      | 
| data           | Genre to use - for data                 | 'comedy'      | str        | 
| train_backward | Choose - train backward/forward model   | False         | bool       |

Run Seq2Seq using:
```
python train_crossent.py -SAVES_DIR <'saves'> -name <'seq2seq'> -BATCH_SIZE <32> -LEARNING_RATE <0.001> -MAX_EPOCHES <100> -TEACHER_PROB <0.5> -data <'comedy'> -train_backward <False>
```

Run backward Seq2Seq using:
```
python train_crossent.py -SAVES_DIR <'saves'> -name <'backward_seq2seq'> -BATCH_SIZE <32> -LEARNING_RATE <0.001> -MAX_EPOCHES <100> -TEACHER_PROB <0.5> -data <'comedy'> -train_backward <True>
```

<p align="center">
  <img src="https://raw.githubusercontent.com/eyalbd2/LanguageModel-UsingRL/master/Images/Training CE Seq2Seq.PNG" width="700" title="Training CE Seq2Seq">
</p> 


#### Train Policy Gradient using BLEU reward
Parameters for calling 'train_rl_BLEU' ( can be seen when writing --help ): 

| Name Of Param     | Description                             | Default Value                                   | Type       | 
| --- | --- | --- | --- | 
| SAVES_DIR         | Save directory                          | 'saves'       									| str        | 
| name              | Specific model saves directory          | 'RL_BLUE'    									| str        | 
| BATCH_SIZE        | Batch Size for training                 | 16            									| int        | 
| LEARNING_RATE     | Learning Rate                           | 1e-4          									| float      | 
| MAX_EPOCHES       | Number of training iterations           | 10000         									| int        | 
| data              | Genre to use - for data                 | 'comedy'      									| str        | 
| num_of_samples    | Number of samples per per each example  | 4                                               | int        | 
| load_seq2seq_path | Pre-trained seq2seq model location      | 'Final_Saves/seq2seq/epoch_090_0.800_0.107.dat' | str        | 

Run using:
```
python train_rl_BLEU.py -SAVES_DIR <'saves'> -name <'RL_BLUE'> -BATCH_SIZE <16> -LEARNING_RATE <0.0001> -MAX_EPOCHES <10000> -data <'comedy'> -num_of_samples <4> -load_seq2seq_path <'Final_Saves/seq2seq/epoch_090_0.800_0.107.dat'>
``` 

<p align="center">
  <img src="https://raw.githubusercontent.com/eyalbd2/LanguageModel-UsingRL/master/Images/Training BLEU RL.PNG" width="500" title="Training BLEU RL">
</p>

#### Train Policy Gradient using MMI reward 
Parameters for calling 'train_rl_MMI' ( can be seen when writing --help ): 

| Name Of Param       | Description                             				  | Default Value                                  			 | Type       | 
| --- | --- | --- | --- | 
| SAVES_DIR           | Save directory                          				  | 'saves'       								   			 | str        | 
| name                | Specific model saves directory          				  | 'RL_Mutual'									   			 | str        | 
| BATCH_SIZE          | Batch Size for training                 				  | 32            								   			 | int        | 
| LEARNING_RATE       | Learning Rate                           				  | 1e-4          								  			 | float      | 
| MAX_EPOCHES         | Number of training iterations           				  | 10000         								   			 | int        | 
| CROSS_ENT_PROB      | Probability to run a CE batch           				  | 0.3           								   			 | float      | 
| TEACHER_PROB        | Probability to run an imitation batch in case of using CE | 0.8           								   		     | float      | 
| data                | Genre to use - for data                 				  | 'comedy'      								    		 | str        | 
| num_of_samples      | Number of samples per per each example  				  | 4                                               		 | int        | 
| load_seq2seq_path   | Pre-trained seq2seq model location      			      | 'Final_Saves/seq2seq/epoch_090_0.800_0.107.dat'          | str        | 
| laod_b_seq2seq_path | Pre-trained backward seq2seq model location 	    	  | 'Final_Saves/backward_seq2seq/epoch_080_0.780_0.104.dat' | str        | 

Run using:
```
python train_rl_MMI.py -SAVES_DIR <'saves'> -name <'RL_Mutual'> -BATCH_SIZE <32> -LEARNING_RATE <0.0001> -MAX_EPOCHES <10000> -CROSS_ENT_PROB <0.3> -TEACHER_PROB <0.8> -data <'comedy'> -num_of_samples <4> -load_seq2seq_path <'Final_Saves/seq2seq/epoch_090_0.800_0.107.dat'> -laod_b_seq2seq_path <'Final_Saves/backward_seq2seq/epoch_080_0.780_0.104.dat'>
``` 

<p align="center">
  <img src="https://raw.githubusercontent.com/eyalbd2/LanguageModel-UsingRL/master/Images/Training RL MMI.PNG" width="500" title="Training RL MMI">
</p>

#### Train Policy Gradient using Perplexity reward 
Parameters for calling 'train_rl_PREPLEXITY' ( can be seen when writing --help ): 

| Name Of Param       | Description                             				  | Default Value                                  	| Type       | 
| --- | --- | --- | --- | 
| SAVES_DIR           | Save directory                          				  | 'saves'       								   	| str        | 
| name                | Specific model saves directory          				  | 'RL_PREPLEXITY'									| str        | 
| BATCH_SIZE          | Batch Size for training                 				  | 32            								   	| int        | 
| LEARNING_RATE       | Learning Rate                           				  | 1e-4          								  	| float      | 
| MAX_EPOCHES         | Number of training iterations           				  | 10000         								   	| int        | 
| CROSS_ENT_PROB      | Probability to run a CE batch           				  | 0.5           								   	| float      | 
| TEACHER_PROB        | Probability to run an imitation batch in case of using CE | 0.5           								   	| float      | 
| data                | Genre to use - for data                 				  | 'comedy'      								    | str        | 
| num_of_samples      | Number of samples per per each example  				  | 4                                               | int        | 
| load_seq2seq_path   | Pre-trained seq2seq model location      			      | 'Final_Saves/seq2seq/epoch_090_0.800_0.107.dat' | str        | 

Run using:
```
python train_rl_PREPLEXITY.py -SAVES_DIR <'saves'> -name <'RL_PREPLEXITY'> -BATCH_SIZE <32> -LEARNING_RATE <0.0001> -MAX_EPOCHES <10000> -CROSS_ENT_PROB <0.5> -TEACHER_PROB <0.5> -data <'comedy'> -num_of_samples <4> -load_seq2seq_path <'Final_Saves/seq2seq/epoch_090_0.800_0.107.dat'>
```

<p align="center">
  <img src="https://raw.githubusercontent.com/eyalbd2/LanguageModel-UsingRL/master/Images/Training RL Perplexity.PNG" width="500" title="Training RL Perplexity">
</p>

#### Train Policy Gradient using Cosine Similarity reward 
Parameters for calling 'train_rl_cosine' ( can be seen when writing --help ): 

| Name Of Param     | Description                             | Default Value                                   | Type       | 
| --- | --- | --- | --- | 
| SAVES_DIR         | Save directory                          | 'saves'       									| str        | 
| name              | Specific model saves directory          | 'RL_COSINE'    									| str        | 
| BATCH_SIZE        | Batch Size for training                 | 16            									| int        | 
| LEARNING_RATE     | Learning Rate                           | 1e-4          									| float      | 
| MAX_EPOCHES       | Number of training iterations           | 10000         									| int        | 
| data              | Genre to use - for data                 | 'comedy'      									| str        | 
| num_of_samples    | Number of samples per per each example  | 4                                               | int        | 
| load_seq2seq_path | Pre-trained seq2seq model location      | 'Final_Saves/seq2seq/epoch_090_0.800_0.107.dat' | str        | 

Run using:
```
python train_rl_cosine.py -SAVES_DIR <'saves'> -name <'RL_COSINE'> -BATCH_SIZE <16> -LEARNING_RATE <0.0001> -MAX_EPOCHES <10000> -data <'comedy'> -num_of_samples <4> -load_seq2seq_path <'Final_Saves/seq2seq/epoch_090_0.800_0.107.dat'>
```

<p align="center">
  <img src="https://raw.githubusercontent.com/eyalbd2/LanguageModel-UsingRL/master/Images/Training RL Cosine.PNG" width="500" title="Training RL Cosine">
</p>

### Testing all models
There are two types of test: automatic and qualitative.

For qualitative test, it is recommended to use the 'use_model' function with pycharm interface. I have suggested (can be seen in code) many sentences to use as source sentences. 

For automatic test, a user can call 'test_all_modells' function, with the parameters:

| Name Of Param       | Description                             				  | Default Value                                  			 | Type       | 
| --- | --- | --- | --- | 
| SAVES_DIR           | Save directory                          				  | 'saves'       								   			 | str        | 
| BATCH_SIZE          | Batch Size for training                 				  | 32            								   			 | int        | 
| data                | Genre to use - for data                 				  | 'comedy'      								    		 | str        | 
| load_seq2seq_path   | Pre-trained seq2seq model location      			      | 'Final_Saves/seq2seq/epoch_090_0.800_0.107.dat' 		 | str        | 
| laod_b_seq2seq_path | Pre-trained backward seq2seq model location 	    	  | 'Final_Saves/backward_seq2seq/epoch_080_0.780_0.104.dat' | str        | 
| bleu_model_path     | Pre-trained BLEU model location      	       		      | 'Final_Saves/RL_BLUE/bleu_0.135_177.dat'                 | str        | 
| mutual_model_path   | Pre-trained MMI model location          			      | 'Final_Saves/RL_Mutual/epoch_180_-4.325_-7.192.dat'      | str        | 
| prep_model_path     | Pre-trained Perplexity model location      			      | 'Final_Saves/RL_Perplexity/epoch_050_1.463_3.701.dat'    | str        | 
| cos_model_path      | Pre-trained Cosine Similarity model location		      | 'Final_Saves/RL_COSINE/cosine_0.621_03.dat'              | str        | 


Run using:
```
python test_all_modells.py -SAVES_DIR <'saves'> -BATCH_SIZE <16> -data <'comedy'> -load_seq2seq_path <'Final_Saves/seq2seq/epoch_090_0.800_0.107.dat'> -laod_b_seq2seq_path <'Final_Saves/backward_seq2seq/epoch_080_0.780_0.104.dat'> -bleu_model_path <'Final_Saves/RL_BLUE/bleu_0.135_177.dat'> -mutual_model_path <'Final_Saves/RL_Mutual/epoch_180_-4.325_-7.192.dat'> -prep_model_path <'Final_Saves/RL_Perplexity/epoch_050_1.463_3.701.dat'> -cos_model_path <'Final_Saves/RL_COSINE/cosine_0.621_03.dat'>
```


## Results for tests
WE present below a table with automatic tests results on 'comedy' movies. For more results and conclusions, please review 'Deep Reinforcement Learning For Language Models.pdf' file in this directory.

<p align="center">
  <img src="https://raw.githubusercontent.com/eyalbd2/LanguageModel-UsingRL/master/Images/automatic evaluations.PNG" width="500" title="Automatic Evaluations">
</p>

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.



## Authors

* **Eyal Ben David** 


## License

This project is licensed under the MIT License 

## Acknowledgments

* Inspiration for using RL in text tasks - "Deep Reinforcement Learning For Dialogue Generation", (Ritter et al., 1996)  
* Policy Gradient Implementation basics in dialogue agents - "Deep Reinforcement Learning Hands-On", by Maxim Lapan (Chapter 12)