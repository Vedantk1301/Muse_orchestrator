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
    # Pass history excluding the last message (which is the current user message)
    generator = run_conversation_stream(user_id, last_user_msg, thread_id, history[:-1])

    # 1. Get the ACK (first chunk)
    try:
        ack_text = await anext(generator)
    except StopAsyncIteration:
        ack_text = "..."

    # Append ACK to history
    history.append({"role": "assistant", "content": ack_text})
    yield history, None, None

    await asyncio.sleep(0.5)

    # 2. Prepare for the Real Response (New Bubble)
    # Start with a typing indicator
    history.append({"role": "assistant", "content": "..."})
    yield history, None, gr.update(visible=True)
    
    full_response = ""
    products = []
    options = []
    import json
    
    async for chunk in generator:
        # Check for hidden product data
        if chunk.startswith("__PRODUCTS__"):
            try:
                products_json = chunk.replace("__PRODUCTS__", "")
                products = json.loads(products_json)
            except Exception:
                pass
            continue
            
        if chunk.startswith("__OPTIONS__"):
            try:
                options_json = chunk.replace("__OPTIONS__", "")
                options = json.loads(options_json)
            except Exception:
                pass
            continue
            
        full_response += chunk
        # Update the LAST message (the real response)
        history[-1]["content"] = full_response
        yield history, None, gr.update(visible=True)

    # After stream ends, if we have products, show them
    if products:
        # Format for Gallery: list of (image_url, caption) tuples
        # Ensure we have valid images
        gallery_data = []
        for p in products:
            img = p.get("image_url") or p.get("image") or "https://via.placeholder.com/300?text=No+Image"
            name = p.get("title") or p.get("name") or "Product"
            price = p.get("price_inr") or p.get("price")
            caption = f"{name}\n(₹{price})" if price else name
            gallery_data.append((img, caption))
            
        yield history, gallery_data, gr.update(visible=True)
        
    # If options found, update the dataset
    if options:
        # Dataset expects a list of lists: [['Option 1'], ['Option 2']]
        samples = [[opt] for opt in options]
        yield history, None, gr.update(samples=samples, visible=True)
    else:
        yield history, None, gr.update(samples=[], visible=False)

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

    # Interactive Chips (Suggestions)
    suggestion_chips = gr.Dataset(
        components=[gr.Textbox(visible=False)],
        label="Quick Replies",
        samples=[],
        visible=False,
    )

    # Product Gallery
    with gr.Row():
        product_gallery = gr.Gallery(
            label="Recommended Products",
            show_label=True,
            elem_id="product_gallery",
            columns=[4],
            rows=[1],
            object_fit="contain",
            height="auto",
        )

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
        [chatbot, product_gallery, suggestion_chips]
    )
    
    submit_btn.click(
        add_user_message, 
        [msg, chatbot], 
        [msg, chatbot]
    ).then(
        bot_response,
        [chatbot, user_id_box],
        [chatbot, product_gallery, suggestion_chips]
    )
    
    # Handle chip clicks
    def on_chip_click(sample):
        # sample is a list like ['Option 1']
        text = sample[0]
        return text

    suggestion_chips.click(
        on_chip_click,
        inputs=[suggestion_chips],
        outputs=[msg]
    ).then(
        add_user_message,
        [msg, chatbot],
        [msg, chatbot]
    ).then(
        bot_response,
        [chatbot, user_id_box],
        [chatbot, product_gallery, suggestion_chips]
    )

    # Status & Preloading
    status_box = gr.Markdown("⏳ Connecting to Muse services...", visible=True)

    async def start_up():
        from responses_agent import Services
        try:
            await Services.ensure_loaded()
            return "✅ Muse Services Ready"
        except Exception as e:
            return f"❌ Service Error: {e}"

    demo.load(start_up, outputs=status_box)

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        show_error=True,
    )
