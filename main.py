import os
from datetime import datetime
import gradio as gr
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama
from crewai_tools import SerperDevTool, tool

load_dotenv()

OPENHERMES_MODEL = "openchat"
CODELLAMA_MODEL = "codellama"

openhermes_llm = Ollama(model=OPENHERMES_MODEL)
search_tool = SerperDevTool()

@tool("Get Today's Date")
def date_tool() -> str:
    """Returns the current date"""
    return datetime.now().strftime("%Y-%m-%d")

researcher = Agent(
    role='Researcher',
    goal='Research and provide accurate information',
    backstory='You are an expert researcher who searches the web for answers to questions',
    tools=[search_tool],
    llm=openhermes_llm,
    human_input=True,
    verbose=True,
    max_iter=5,
    allow_delegation=False
)

def process_query(query):
    search_task = Task(
        description=f'Question provided: {query}',
        expected_output='Respond to the question with accurate information',
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[search_task],
        verbose=1,
        process=Process.sequential
    )

    return crew.kickoff()

def create_gradio_interface():
    with gr.Blocks() as iface:
        gr.Markdown("## Openchat Chatbot")
        chatbot = gr.Chatbot()
        query = gr.Textbox(placeholder="Enter your query")
        state = gr.State([])

        def user(user_message, chat_history):
            chat_history.append([user_message, None])
            return "", chat_history

        def bot(chat_history):
            user_message = chat_history[-1][0]
            chat_history[-1][1] = "Searching the web..."
            chatbot.value = chat_history
            bot_message = process_query(user_message)
            chat_history[-1][1] = bot_message
            return chat_history

        query.submit(user, [query, state], [query, chatbot], queue=False).then(
            bot, state, chatbot
        )
        query.submit(lambda: "", None, query)

    return iface

if __name__ == '__main__':
    iface = create_gradio_interface()
    iface.launch()