https://colab.research.google.com/drive/10x7cG7Z3vHv0Az5sxn5XQ_L3gc3H8ew5?authuser=1#scrollTo=3YddfMp-DpQK
# Naive_Bayes_for_Polarity_Analysis

This work presents a survey of a comparative study between Multinomial and Bernoulli Naïve Bayes algorithms for text classification, with a focus on news polarity analysis.
To ensure reproducibility and practicality, we present a validated implementation of both algorithms for news and movie's review polarity analysis. This implementation allows researchers and practitioners to reproduce the experiments and apply the models to new datasets or variations. By offering a clear comparison and implementation guidance, our study enables informed decision-making when selecting the appropriate algorithm for text classification tasks, particularly in the context of news polarity analysis.

# Impementation
The project includes several components that train and evaluate Naïve Bayes models on different datasets. It starts with importing necessary libraries, including pandas, numpy, matplotlib, nltk, transformers, and scikit-learn modules. The code then defines functions for data augmentation and text generation, which are later used for creating fake datasets. The code collects three datasets and stores them in a dataframe, making them ready for subsequent training and testing of the models. The next part of the code preprocesses the text in each dataset by converting it to lowercase, tokenizing it, removing punctuation, and eliminating stopwords using nltk. Then, it vectorizes the preprocessed text using CountVectorizer from scikit-learn. The datasets are split into training and testing sets for further evaluation. Finally, the code compares the accuracy of the two Naïve Bayes models using ROC curves and bar charts, providing a visual comparison of their performance.

# Mathematical Definition
Let us define a news classification problem as follows: Given a set of news articles, denoted as \(X = \{x_1, x_2, ..., x_n\}\), where each news article \(x_i\) is represented by a feature vector containing relevant textual and/or metadata information, the objective is to assign a predefined polarity label \(y_i\) from a set of -1 for negative sentences and 1 for the positive ones to each news article \(x_i\). The problem can be formulated as a function \(f: X \rightarrow Y\), where \(f(x_i) = y_i\) represents the mapping from the input news article \(x_i\) to its corresponding polarity label \(y_i\). The task of news classification is to learn an effective classification model or algorithm that can accurately predict the polarity label for new, unseen news articles based on the available training data.

# Datasets
In this project, we leverage the power of three distinct datasets to drive our analysis and research. Each dataset contributes unique perspectives and valuable insights, enhancing the depth and scope of our work. The first dataset comprises movie reviews. The second dataset is centered around news articles. In addition to these datasets, we have generated our own synthetic dataset specifically designed for sentiment analysis and text classification.

### Plot of the News Article dataset
![News_article](https://github.com/DelaramRajaei/Naive_Bayes_for_Polarity_Analysis/assets/48606206/1b5645cd-c386-4e67-8320-0904e4a7aaf5)

### Plot of the Movie Review dataset
![Movie_review](https://github.com/DelaramRajaei/Naive_Bayes_for_Polarity_Analysis/assets/48606206/9989f260-a2ec-41d9-b708-3f8684378239)

# Results
For this project, we employ two distinct models, namely the Multinomial model and the Bernoulli model, to analyze three different datasets. The choice between these models for polarity analysis is contingent upon the specific attributes of the data and the requirements of the sentiment analysis task being conducted. Upon analyzing the results, we find that both models yield comparable accuracy levels. The Bernoulli model, in particular, slightly outperforms the Multinomial model across the board. However, the differences in performance are relatively subtle. Overall, both models demonstrate effective capabilities in predicting sentiment polarity within the datasets.

### Comparing two models for Movie Review dataset
![movie_compare](https://github.com/DelaramRajaei/Naive_Bayes_for_Polarity_Analysis/assets/48606206/df83958e-4f84-4776-9cda-31394b25091d)

![Results_movie](https://github.com/DelaramRajaei/Naive_Bayes_for_Polarity_Analysis/assets/48606206/4751268a-10ee-415c-90b4-fa7a0714d271)


