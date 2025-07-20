import gradio as gr
from openai import OpenAI
from CONSTANTS import API_KEY, MODEL_NAME, BASE_URL

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def predict(message, history):
    # Format history for OpenAI API
    history_openai_format = []
    if history and len(history) > 1:
        for each in history:
            history_openai_format.append(each)
    if message:
        history_openai_format.append({"role": "user", "content": message})

    response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=history_openai_format,
    temperature=1.0,
    stream=True
)
    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            partial_message += chunk.choices[0].delta.content
            yield partial_message

# Extend the chatbot height for infinite scroll style
custom_css = """
#chatbot { height: 80vh !important; overflow-y: auto !important; }
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("## OpenAI Streaming Chatbot with Infinite Scroll")
    chatbot = gr.Chatbot(elem_id="chatbot", type="messages")
    chat_interface = gr.ChatInterface(
        predict,
        type="messages",
        chatbot=chatbot
    )

demo.queue()
demo.launch()
