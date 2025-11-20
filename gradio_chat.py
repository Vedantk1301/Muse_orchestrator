# app/gradio.py

import gradio as gr
import asyncio
from responses_agent import run_conversation_stream

async def add_user_message(user_message, history):
    """
    Append the user message to history and clear the textbox.
    """
    if not user_message.strip():
        return "", history
    return "", history + [{"role": "user", "content": user_message}]

async def bot_response(history, user_id):
    """
    Generator that:
    1. Calls the agent stream.
    2. Yields the ACK as a separate assistant message.
    3. Yields the final response as ANOTHER separate assistant message.
    """
    if not history:
        return

    last_user_msg = history[-1]["content"]
    thread_id = f"webui-{user_id}"

    # Create the generator
    generator = run_conversation_stream(user_id, last_user_msg, thread_id)

    # 1. Get the ACK (first chunk)
    try:
        ack_text = await anext(generator)
    except StopAsyncIteration:
        ack_text = "..."

    # Append ACK to history
    history.append({"role": "assistant", "content": ack_text})
    yield history

    # Small delay to let the user register the ACK
    await asyncio.sleep(0.5)

    # 2. Prepare for the Real Response (New Bubble)
    # Start with a typing indicator
    history.append({"role": "assistant", "content": "..."})
    yield history
    
    full_response = ""
    async for chunk in generator:
        full_response += chunk
        # Update the LAST message (the real response)
        history[-1]["content"] = full_response
        yield history

with gr.Blocks(title="🎨 MuseBot — Fashion Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎨 MuseBot — Fashion Assistant
        Ask me for outfits, pieces, or styling tips.
        """
    )
    
    chatbot = gr.Chatbot(
        type="messages", 
        label="MuseBot", 
        height=600,
        avatar_images=(None, "https://cdn-icons-png.flaticon.com/512/4712/4712035.png") # Optional avatar
    )
    
    with gr.Row():
        msg = gr.Textbox(
            scale=4, 
            placeholder="Ask me something fashion-related...", 
            label="Your Message",
            autofocus=True
        )
        user_id_box = gr.Textbox(value="demo_user", label="User ID", scale=1)
        submit_btn = gr.Button("Send", variant="primary", scale=1)

    # Examples
    gr.Examples(
        examples=[
            "Hey! I'm looking for a blue cotton shirt",
            "What goes well with black jeans?",
            "Show me some summer outfits",
            "I want striped shirts for a beach vacation",
        ],
        inputs=msg
    )

    # Event Wiring
    # 1. User submits -> add message to chat -> clear input
    msg.submit(
        add_user_message, 
        [msg, chatbot], 
        [msg, chatbot]
    ).then(
        # 2. Bot responds (streaming)
        bot_response,
        [chatbot, user_id_box],
        [chatbot]
    )
    
    submit_btn.click(
        add_user_message, 
        [msg, chatbot], 
        [msg, chatbot]
    ).then(
        bot_response,
        [chatbot, user_id_box],
        [chatbot]
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )
