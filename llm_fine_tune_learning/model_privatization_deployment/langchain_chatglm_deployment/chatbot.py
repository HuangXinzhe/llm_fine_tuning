"""
通过langchain使用ChatGLM3模型进行对话
需要先启动ChatGLM3模型服务
以ChatGLM3模型为例，启动ChatGLM3模型服务的命令如下：
在ChatGLM3模型目录下找到api_server.py文件,执行命令：
python api_server.py
"""

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.messages import AIMessage
from langchain_community.llms.chatglm3 import ChatGLM3

template = """{question}"""
prompt = PromptTemplate.from_template(template)

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

llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "北京和上海两座城市有什么不同？"

llm_chain.run(question)
