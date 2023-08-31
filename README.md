# Dissertation_sc22av
Dissertation for Advanced Computer Science 2023

1.  Project : Deep Dive into Sentiment: Comparative Study of language models across domain specific datasets
2.  Installation & Setup:

Pre-requisites:
The models have been run on different environments which are on different chip sets. The setup needs to be similar to be able to run the code without any issues.
Dataset_1_FASTText.ipynb ,
Dataset_1_FINBERT.ipynb - 
These run on enivornment-1 with platform macOS-10.16-x86_64-i386-64bit and python version 3.9.12. 
Dataset_1_Lexicon.ipynb ,
Dataset_1_LSTM.ipynb ,
Dataset_1_SVM.ipynb ,
Dataset_1_Naive Bayes.ipynb -
These run on enivornment-1 with platform macOS-13.5-arm64-i386-64bit and python version 3.11.4.
The code has been executed on jupyter lab JupyterLab 3.3.2.

The models have been run on different environments which are on different chip sets. The setup needs to be similar to be able to run the code without any issues.
Dataset_2_FASTText.ipynb -
These run on enivornment-1 with platform macOS-10.16-x86_64-i386-64bit and python version 3.9.12.
Dataset_2_BERTweet.ipynb - (mode.to('cpu') - not compartible with mps) ,
Dataset_2_Lexicon.ipynb ,
Dataset_2_LSTM.ipynb ,
Dataset_2_SVM.ipynb ,
Dataset_2_Naive Bayes.ipynb -
These run on enivornment-1 with platform macOS-13.5-arm64-i386-64bit and python version 3.11.4.


Dependencies for Dataset-1 & Dataset-2

Packages to be Downloaded:
numpy,
pandas,
nltk (Natural Language Toolkit),
BeautifulSoup (from bs4),
textblob,
matplotlib,
seaborn,
scikit-learn (or sklearn),
gensim,
tensorflow (and potentially tensorflow_hub, tensorflow_text if you plan to uncomment them),
keras_tuner (for hyperparameter tuning with TensorFlow/Keras),
fasttext,
transformers (from Hugging Face),
torch (PyTorch),
optuna,
contractions,

Installation:-(#you can use following pip command in jupyter notebook and execute to install packages.) :-  
!pip install numpy pandas nltk beautifulsoup4 textblob matplotlib seaborn scikit-learn gensim tensorflow keras_tuner fasttext transformers torch optuna contractions


Additionally, for some NLP tasks in code, ensure you've downloaded the necessary nltk packages
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

Dataset Setup 

Download the dataset-1 from https://www.kaggle.com/datasets/ankurzing/aspect-based-sentiment-analysis-for-financial-news 
Download the dataset-2 from HTTPS://WWW.KAGGLE.COM/DATASETS/YASSERH/TWITTER-TWEETS-SENTIMENT-DATASET 
Once downloaded, please change the data_pth in all the .ipynb files based on the path of your file directory 

3. Once this is setup you can run the code using jupyter lab.

4.
For dataset-1
Dataset_1_FASTText - has implementation of FastText,
Dataset_1_FINBERT - has implementation of FinBERT,
Dataset_1_Lexicon - has implementation of VADER and TextBlob,
Dataset_1_LSTM - has implementation of standard LSTM,Stacked LSTM and BiLSTM,
Dataset_1_SVM - has implementation of SVM with TF-IDF and GloVe embedding(it has been added to the git repo),
Dataset_1_Naive Bayes - has implementation of variants of Naive Bayes

For dataset-2 
Dataset_2_FASTText - has implementation of FastText,
Dataset_2_BERTweet - has implementation of BERTweet,
Dataset_2_Lexicon - has implementation of VADER and TextBlob,
Dataset_2_LSTM - has implementation of standard LSTM,Stacked LSTM and BiLSTM,
Dataset_2_SVM - has implementation of SVM with TF-IDF and GloVe embedding(it has been added to the git repo),
Dataset_2_Naive Bayes - has implementation of variants of Naive Bayes,

6. There is a functionality of getting metrics which needs to be done manually -
   
   Dataset_1_FASTText,Dataset_1_FINBERT,Dataset_1_Lexicon,Dataset_1_LSTM - for these to get specificity and FPR - matrix value need to be added manually to calculate and print the same. 
   Dataset_2_FASTText,Dataset_2_BERTweet,Dataset_1_Lexicon,Dataset_1_LSTM - for these to get specificity and FPR - matrix value need to be added manually to calculate and print the same.
   
example :- #need to enter the values to the matrix manually as this can change based on the fine tuning of the model
cm = np.array([[281, 49, 23], 
               [33, 328, 28], 
               [37, 79, 321]])

compute_FPR_spec_metrics(cm)
-x-x-
