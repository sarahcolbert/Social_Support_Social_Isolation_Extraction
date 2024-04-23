# Using Natural Language Processing to Identify  Social Support and Social Isolation from Electronic Health Records of Psychiatric Patients: A Multi-Site Study

This repository maintains the code for identifying social support and social isolation with fine-grained categories (presence or absence of social network, emotional support, instrumental support, general and loneliness) using rule- and large language models (LLM)-based algorithms. 

## 1. For running rule-based algorithm
	python rule_based_classification.py

## 2. For running LLM 

### for fine-tunning the models
	python llm_fine_tune_all.py

### for testing the models
	python llm_sentence_classification_fine_tuned.py

We will update the GPU version of the LLM code soon. 


## Reference

Please cite the following paper if you find this code useful.

@misc{patra2024extracting,
      title={Extracting Social Support and Social Isolation Information from Clinical Psychiatry Notes: Comparing a Rule-based NLP System and a Large Language Model}, 
      author={Braja Gopal Patra and Lauren A. Lepow and Praneet Kasi Reddy Jagadeesh Kumar and Veer Vekaria and Mohit Manoj Sharma and Prakash Adekkanattu and Brian Fennessy and Gavin Hynes and Isotta Landi and Jorge A. Sanchez-Ruiz and Euijung Ryu and Joanna M. Biernacka and Girish N. Nadkarni and Ardesheer Talati and Myrna Weissman and Mark Olfson and J. John Mann and Alexander W. Charney and Jyotishman Pathak},
      year={2024},
      eprint={2403.17199},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

Details will be provided soon. [link](https://github.com/brajagopalcse/SISU)

Please contact brajagopal[dot]cse[at]gmail[dot]com for further information. 

