import gradio as gr
from langchain import PromptTemplate, LLMChain
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint
from transformers import pipeline
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"]
pipe = pipeline(
    'text2text-generation',
    model='VMware/flan-t5-large-alpaca',
    max_length=60,
    do_sample=True,
    temperature=0.9
)
llm = HuggingFacePipeline(pipeline=pipe)
prompt_template = """ you are a highly knownlegdable and helpful AI assistant.Engage in a conversation with the user.you are a responsible AI Assistant so answer the questions acccording to your best knowlegde.your main is to provide clear and informative answer to the user questions.
User: {question}
Assistant:"""
prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
chain = LLMChain(llm=llm, prompt=prompt)
def chatbot(question,chat_history):
    response = chain.run(question)
    return response
demo = gr.ChatInterface(
    fn=chatbot,
    #inputs=gr.Textbox(label="Question"),
    #outputs=gr.Textbox(label="Answer"),
    title="Chatbot",
    description="Helpful AI Assistant!!"
)
demo.launch()