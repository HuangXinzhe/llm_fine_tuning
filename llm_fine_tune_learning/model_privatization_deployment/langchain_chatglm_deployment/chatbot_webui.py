import gradio as gr

from langchain.schema.messages import AIMessage
from langchain_community.llms.chatglm3 import ChatGLM3
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

CHATGLM_URL = "http://127.0.0.1:8000/v1/chat/completions"

def init_chatbot():

    messages = [
        AIMessage(content="欢迎问我任何问题。"),
    ]

    llm = ChatGLM3(
        endpoint_url=CHATGLM_URL,
        max_tokens=80000,
        prefix_messages=messages,
        top_p=0.9,
        model_kwargs={"sample_model_args": False},
    )

    global CHATGLM_CHATBOT
    CHATGLM_CHATBOT = ConversationChain(llm=llm, 
                                        verbose=True,
                                        memory=ConversationBufferMemory())
    return CHATGLM_CHATBOT

def chatglm_chat(message, history):
    ai_message = CHATGLM_CHATBOT.predict(input = message)
    return ai_message

def launch_gradio():
    demo = gr.ChatInterface(
        fn=chatglm_chat,
        title="ChatBot (Powered by ChatGLM)",
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    # 初始化聊天机器人
    init_chatbot()
    # 启动 Gradio 服务
    launch_gradio()
