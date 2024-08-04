from groq import Groq
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import json
import models
import os


user_input_buffer = []
ai_output_queue = []

chat_ai = models.get_groq_llama3_groq_70b_tool_use_preview()

# open log file for write 
def conversation_log(text):
    with open('logs/conversation_log.txt', 'a') as f:



def processing_loop():
    messages = []

    while True:
        if not user_input_buffer.empty():
            user_input = user_input_buffer.get()
            messages.append(HumanMessage(content=user_input))

        if messages:
            ai_response = chat_ai(messages)
            messages.append(AIMessage(content=ai_response.content))
            ai_output_queue.put(ai_response.content)
    
def user_input_handler():
    while True:
        user_input = input('USER: ')
        user_input_buffer.put(user_input)

# def display():
#     while True:
#         if not ai_output_queue.empty():
#             ai_response = ai_output_queue.get()
#             print('AI: ' + ai_response)


