### **Introduction**
This git repository is created for Covid-19 Sentiment Analysis project. This project implementation consists of the following methods:
1. A data pre-processing methodology to get rid of symbols and stopwords.
2. A convolutional neural network (CNN) to train the model to classify social media posts based on categories ("medical research", "social events", "pandemic data", "administrative policy").
3. Accuracy plots for each epoch 

### **Software requirements**
1. Python 3.7 or above
2. Python IDE (Visual Studio Code)

### **Standard CPU-based implementation**

#### **I - Clone the repository**

#### **II - Set up the virtual environment**
1. Create a new folder and name it as COVID19_SentimentAnalysis.
2. Copy news_classification.py from cloned repository and place it inside COVID19_SentimentAnalysis folder.
3. Download [GoogleNews-vectors-negative300.bin](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g). This will need to b e unzipped. Then place the unzipped .bin file inside the COVID19_SentimentAnalysis folder.
4. Download the training_data_set_unduplicate.csv from the clones repository and place it inside the COVID19_SentimentAnalysis folder.
5. In Visual Studio Code, go to Terminal and run the below in cmd terminal. This creates a virtual environment called "sentimentanalysis-venv"
   
            python -m venv sentimentanalysis-venv
5. For Windows, run below line to activate the virtual environment
   
            sentimentanalysis-venv\Scripts\activate.bat

   
#### **III - Install python packages**

            pip install -r requirements.txt        

#### **IV - Run the script**
Now, we are all set to run the script.
