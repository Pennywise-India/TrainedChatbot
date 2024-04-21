import random
import json
import pickle
import numpy as np
import nltk
import streamlit as st
import os
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('D:\create_chatbot_using_python-main\Qna bot\intents.json').read())

words = pickle.load(open('D:\create_chatbot_using_python-main\Qna bot\words.pkl', 'rb'))
classes = pickle.load(open('D:\create_chatbot_using_python-main\Qna bot\classes.pkl', 'rb'))
model = load_model('D:\create_chatbot_using_python-main\Qna bot\models\chatbot_model_trained_till_completed1.h5')
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words (sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class (sentence):
    bow = bag_of_words (sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes [r[0]], 'probability': str(r[1])})
    return return_list
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice (i['responses'])
            break
    return result


st.set_page_config(page_title="Chatbot")

# st.header("")



def load_chat_history():
    chat_history = []
    if os.path.exists("chat_history.txt"):
        with open("chat_history.txt", "r") as file:
            chat_history = file.readlines()
    return chat_history

def save_chat_history(chat_history):
    with open("chat_history.txt", "w") as file:
        file.writelines(chat_history)


def main():
    st.title("Budget Buddy")
    st.text('''Hello! I am designed to help you in getting insights regarding union budget
allocation from 1980 to 2019 in particularly in five sectors[Education,Medical, Defence, 
Railway, Agriculture].''')
    # Load chat history from file
    chat_history = load_chat_history()

    # Display chat history
    # st.subheader("Chat History:")
    # for chat in chat_history:
    #     st.text(chat.strip())

    # Input text box for user message
    message = st.text_input("Ask me anything:", key="input")


    submit = st.button("Reply")

    if message.strip():
        chat_history.append(f"You: {message} \n")

        ints = predict_class (message)
        res = get_response (ints, intents)
        st.write(res)
        chat_history.append(f"DevBot: {res} \n")
        save_chat_history(chat_history)

        st.subheader("Chat History:")
        for chat in chat_history:
            st.text(chat.strip())
        
        if "bye" in message.lower():
            st.stop()
            

if __name__ == "__main__":
    main()





# message=st.text_input("Enter a message: ",key="input")
# submit= st.button("Reply me")
# if submit:
#     ints = predict_class (message)
#     res = get_response (ints, intents)
#     st.write(res)
    

# def main():
#     st.title("Chatbot Interface")

#     # Initialize chat history list
#     chat_history = []

#     # Display chat history
#     st.subheader("Chat History:")
#     for chat in chat_history:
#         st.text(chat)

#     # Input text box for user message
#     message = st.text_input("Enter a message:", key="input")

#     # Button to submit the message
#     submit = st.button("Reply")

#     if submit and message.strip():  # Check if message is not empty
#         # Record user message in chat history
#         chat_history.append(f"You: {message}")

#         # Predict intent and get response
#         ints = predict_class (message)
#         res = get_response (ints, intents)

#         # Record bot response in chat history
#         chat_history.append(f"Bot: {res}")

#         # Display updated chat history
#         st.subheader("Updated Chat History:")
#         for chat in chat_history:
#             st.text(chat)

# if __name__ == "__main__":
#     main()
# print("GO! Bot is running!")

# while True:
#     message = input("")
#     ints = predict_class (message)
#     res = get_response (ints, intents)
#     print (res)
    

