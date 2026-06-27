import streamlit as st
def get_input():
    user_input=input("You:")
    return user_input.lower().strip()

def process_query(user_input,know_base):
    cleaned_input = user_input.lower().strip()
    
    for key in know_base:
        if key in cleaned_input:
            return know_base[key]
    return "Sorry! I do not understand what you are saying."
def get_intents():
    return{"hi": "Hi there! How can I assist you?",
             "name": "My name is Rule-Based Chatbot",
             "morning": "Good Morning to you too",
             "wellbeing":"I am fine.What about you?",
             "eid":"Eid Mubarak to you too. Enjoy"
    }

st.title("Rule-Based ChatBot")
user_input=st.text_input("You:",placeholder="Enter your query")
# no need of while True as Streamlit handles continuous cycles itself 
   
if user_input:

    if user_input in ["exit","quit","q"]:
        st.error("Quitting.....") # instead of break  
    else:
        intents=get_intents()
        response=process_query(user_input,intents)
        st.info(f"{response}")
else:
    st.write("Start Chat")





