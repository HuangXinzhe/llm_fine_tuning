from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema.messages import AIMessage
from langchain_community.llms.chatglm3 import ChatGLM3


endpoint_url = "http://127.0.0.1:8000/v1/chat/completions"

messages = [
    AIMessage(content="我将从美国到中国来旅游，出行前希望了解中国的城市"),
    AIMessage(content="欢迎问我任何问题。"),
]

llm = ChatGLM3(
    endpoint_url=endpoint_url,
    max_tokens=80000,
    prefix_messages=messages,
    top_p=0.9,

)

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

print(conversation.predict(input="你们衣服怎么卖？"))
print(conversation.predict(input="有哪些款式？"))
print(conversation.predict(input="休闲装男款都有啥？"))
