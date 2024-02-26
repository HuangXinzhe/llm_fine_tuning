"""
通过langchain使用ChatGLM
官方文档：https://python.langchain.com/docs/integrations/chat/zhipuai
"""

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

zhipuai_api_key = "your_api_key"

chat = ChatZhipuAI(
    temperature=0.5,
    api_key=zhipuai_api_key,
    model="chatglm_turbo",
)

messages = [
    AIMessage(content="Hi."),
    SystemMessage(content="Your role is a poet."),
    HumanMessage(content="Write a short poem about AI in four lines."),
]

response = chat(messages)
print(response.content)  # Displays the AI-generated poem
