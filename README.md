# CTC2021

## Description
Chinese Text Correction (CTC) is to detect and correct errors in the text, which is an important task of natural language process. In the past, text correction have used texts written by foreign language learners, and most of the errors in these texts are mistakes that native Chinese writers would not make. A proofreading system for native Chinese speakers would be more helpful for government documents, press and publishing industries. Therefore, CTC 2021 selects web texts written by native Chinese writers on the Internet as proofreading data, and examines the cognitive intelligence of the machine in terms of spelling errors, grammatical errors and faulty wording or formulation. Given a text, the participators should build models to detect the wrong words and error types, and correct them. [Competition URL](https://competitions.codalab.org/competitions/32702#learn_the_details)

## Datasets

## Method

* #### GED (Peiyuan Gong) 
    Use NER-based framework for GED task, which is a subtask of GEC, identifying different error start and end index. Here we use `<INS>`, `<REP>` and `<DEL>` three basic labels to represent `缺失`, `别字` and `冗余`。
* #### GEC-NAG (Yinghao Li)
* #### GEC-AG (Haiming Wu)

## Metrics
The detection score and correction score are considered together, specifically, the evaluation result = 0.8 * detection score + 0.2 * correction score, where both the detection score and correction score are calculated using the F-score.

## Result

* #### GED
* #### GEC-NAG
* #### GEC-AG

## Team

* Peiyuan Gong (BIT)
* Yinghao Li (BIT)
* Haiming Wu (BIT)

## Connection

* pygongnlp@gmail.com
* 3120201019@bit.edu.cn
