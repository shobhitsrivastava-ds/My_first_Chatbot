from flask import Flask,render_template, url_for , redirect, flash, request,send_from_directory

import numpy as np
import os

#from tensorflow.keras.np_utils import probas_to_classes
import random
import pandas as pd
import string


import re
import nltk

import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.chat.util import Chat, reflections

app=Flask(__name__,template_folder='template')


nltk.download('punkt')
nltk.download("wordnet")



# RELATED TO THE SQL DATABASE
app.config['SECRET_KEY'] = '4791628bb0b13ce0c676dfde280ba245'
#app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///site.db"
#db=SQLAlchemy(app)

#from model import User,Post

#//////////////////////////////////////////////////////////

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = 'uploads'

""" NEW ADDED TEST"""###############################################################################
data=""
lemmer = nltk.stem.WordNetLemmatizer()
full_name= ""
def read_file(path):
    with open(path,"r") as f:
        text= f.read().strip()
    return(text)
# Function for the preperation of the data
tokenizer = Tokenizer()

def greeting(sentences):
    GREETING_INPUTS = ("hello","hi","greetings","sup","what's")
    GREETING_RESPONSES =["hi","hey","nods*","hi there", "hello", "I am glad! you are talking to me"]
    for word in sentences.split():
        if word.lower() in GREETING_INPUTS:
            return(random.choice(GREETING_RESPONSES))

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Function for generating the response

print("data= {}".format(data))

def response(user_response, sent_tokens):
    robo_response =" "
    sent_tokens.append(user_response)
    #print("Printing the sentence token...")
    #for i in sent_tokens:
    #    print(i)
    #    print("end")
    #print("/////////////////////////////////")
    Tfidf = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = Tfidf.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    #print("vals= {}".format(vals))
    idx=vals.argsort()[0][-2]
    #print("idx= {}".format(idx))
    #print("vals argsort= {}".format(vals.argsort()[0]))
    #print(.format())
    flat = vals.flatten()
    flat.sort()
    #print("flat= {}".format(flat))
    req_tfidf = flat[-2]
    #print("req_tfidf= {}".format(req_tfidf))
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

####################################################################################################
# procesing uploaded file and predict it



@app.route("/upl", methods=['POST'])
def upl():
    try:
        file = request.files['file']
        global full_name
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)
        #print("fname= {}".format(full_name))
        return(render_template('predict.html', reply= ""))
    except:
        flash("Upload the file & then try again", "danger")      
        return redirect(url_for("my_bot"))


@app.route('/upload', methods=['POST','GET'])
def upload_file():
    text=""
    if request.method == 'GET':
        return render_template('index.html')
    else:
        ##file = request.files['file']
        ##full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        ##file.save(full_name)
        #print("Full name={}".format(full_name))
        global data
        data= read_file(full_name)
        data= re.sub('[!#?,:";\n]', '', data)
        print("data= {}".format(data))
        sent_tokens = nltk.sent_tokenize(data)
        word_tokens = nltk.word_tokenize(data)
        #print("len of sent={}".format(len(sent_tokens)))
        wds= str(request.form.get("words"))
        user_resp = wds.lower()
        #print("length of sentence token= {}".format(len(sent_tokens)))
        if(user_resp!="bye"):
            if(user_resp=="thanks" or user_resp=="thanks you"):
                flag=False
                text= "you are welcome"
            elif(user_resp=="how are you?" or user_resp=="what's up?"):
                text= "I am fine, thanks!"
            elif(user_resp=="what are you doing?") :
                text= "Answering your queries !!"
            elif(user_resp=="what can you do?" or user_resp== "what can you do for me?"):
                text= "Answer you Queries :)"
            else:
                if(greeting(user_resp)!=None):
                    text= greeting(user_resp)
                else:
                    text= response(user_resp, sent_tokens)
                    if(user_resp in sent_tokens):
                        sent_tokens.remove(user_resp)
                    #if(user_resp in sent_tokens):
                     #   sent_tokens.remove(user_resp)
                    #sent_tokens.remove(user_resp)
        else:
            flag =False
            text= "Bye! take care.."
    return render_template('predict.html', reply= text)
"""except:
            flash("Upload & Fill the form correctly & then try again", "danger")      
            return redirect(url_for("my_bot"))"""
        #except :
         #   flash("Please select the image first !!", "success")      
          #  return redirect(url_for("clean"))

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/")

@app.route("/home")
def home():
	return render_template("home.html")

@app.route("/about")
def about():
	return render_template("about.html")

@app.route("/demo")
def demo():
    return render_template("predict.html")

@app.route("/my_bot")
def my_bot():
    return render_template("index.html")

if __name__ == "__main__":
	app.run(debug=True)
