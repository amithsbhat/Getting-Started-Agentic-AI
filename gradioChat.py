
import gradio as gr
from openai import OpenAI
from CONSTANTS import API_KEY, MODEL_NAME, BASE_URL


def create_openai_client():
    """Create and return an OpenAI client instance."""
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def format_history_for_openai(message, history):
    """Format the conversation history for the OpenAI API."""
    history_openai_format = []
    if history and len(history) > 1:
        history_openai_format.extend(history)
    if message:
        history_openai_format.append({"role": "user", "content": message})
    return history_openai_format


def predict(message, history):
    """
    Generates a streaming response from an OpenAI chat model based on the provided user message and conversation history.
    Args:
        message (str): The latest user message to be sent to the model.
        history (list): A list of previous messages in the conversation, each formatted as a dictionary with 'role' and 'content' keys.
    Yields:
        str: The progressively generated partial response from the model as it streams.
    Notes:
        - The function formats the conversation history for the OpenAI API.
        - It streams the model's response, yielding partial outputs as they arrive.
    """
    client = create_openai_client()
    history_openai_format = format_history_for_openai(message, history)
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


custom_css = """
#chatbot { height: 80vh !important; overflow-y: auto !important; }
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("## OpenAI Streaming Chatbot with Infinite Scroll")
    chatbot = gr.Chatbot(elem_id="chatbot", type="messages")
    gr.ChatInterface(
        predict,
        type="messages",
        chatbot=chatbot
    )

demo.queue()
demo.launch()
