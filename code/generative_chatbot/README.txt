README GENERATIVE CHATBOT:
Python: 3.0
Tensorflow Version: 1.1v
Libraries Required: nltk, numpy
Data Source: https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

1. Firstly, raw data need to be processed, run data.py provided in data_set folder. Already processed data also provided.
2. Now run chatbot_generative.py to start training the model.
3. After, every 10,000 epochs trained model will be saved in saved_generative_model folder.

High configuration system with good GPU support required to run the code and get output. 
If using --gpu version tensorflow and memory problem occurs reduce the number of rnn layers to one in
chatbot_generative.py  
