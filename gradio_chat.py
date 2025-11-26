# app/gradio.py

import gradio as gr
import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple

from responses_agent import run_conversation_stream, logger as agent_logger


META_CHUNK_PREFIX = "__META__"
LEGACY_PRODUCTS_PREFIX = "__PRODUCTS__"
LEGACY_OPTIONS_PREFIX = "__OPTIONS__"
PLACEHOLDER_IMAGE = "https://via.placeholder.com/300?text=No+Image"

def _parse_stream_chunk(chunk: str) -> Tuple[Optional[str], Any]:
    """
    Inspect a streamed chunk and return (type, payload) for metadata chunks.
    """
    if not isinstance(chunk, str):
        return None, None

    if chunk.startswith(META_CHUNK_PREFIX):
        raw = chunk[len(META_CHUNK_PREFIX):]
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            agent_logger.warning("UI meta chunk decode failed", error=str(exc))
            return None, None
        return payload.get("type"), payload.get("data")

    if chunk.startswith(LEGACY_PRODUCTS_PREFIX):
        raw = chunk[len(LEGACY_PRODUCTS_PREFIX):]
        try:
            return "products", json.loads(raw)
        except json.JSONDecodeError as exc:
            agent_logger.warning("Legacy products chunk decode failed", error=str(exc))
            return None, None

    if chunk.startswith(LEGACY_OPTIONS_PREFIX):
        raw = chunk[len(LEGACY_OPTIONS_PREFIX):]
        try:
            return "options", json.loads(raw)
        except json.JSONDecodeError as exc:
            agent_logger.warning("Legacy options chunk decode failed", error=str(exc))
            return None, None

    return None, None


def _format_gallery_data(products: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    Convert product dicts into gallery tuples expected by gr.Gallery.
    """
    gallery_data: List[Tuple[str, str]] = []
    for product in products:
        if not isinstance(product, dict):
            continue
        image = (
            product.get("image_url")
            or product.get("image")
            or PLACEHOLDER_IMAGE
        )
        name = product.get("title") or product.get("name") or "Product"
        price = product.get("price_inr")
        if price in (None, "", 0):
            price = product.get("price")
        caption = name
        if price not in (None, "", 0):
            caption = f"{name}\n(Rs. {price})"
        gallery_data.append((image, str(caption)))
    return gallery_data


def _format_option_samples(options: List[Any]) -> List[List[str]]:
    """
    Prepare dataset samples for clickable chips.
    """
    samples: List[List[str]] = []
    for option in options:
        if option is None:
            continue
        text = str(option).strip()
        if not text:
            continue
        samples.append([text])
    return samples


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

    # 1. Get the first chunk to check for ACK
    try:
        first_chunk = await anext(generator)
    except StopAsyncIteration:
        return

    # Check if it's an ACK
    if isinstance(first_chunk, str) and first_chunk.startswith("__ACK__"):
        ack_text = first_chunk.replace("__ACK__", "")
        # Yield ACK bubble
        history.append({"role": "assistant", "content": ack_text})
        yield history, None, None
        
        await asyncio.sleep(0.2)
        
        # Prepare for main response bubble
        history.append({"role": "assistant", "content": "..."})
        yield history, None, None
        
        # Consume next chunk for main response
        try:
            first_chunk = await anext(generator)
        except StopAsyncIteration:
            pass
    else:
        # No ACK, start main response immediately
        history.append({"role": "assistant", "content": "..."})
        yield history, None, None

    # Reset UI elements
    gallery_reset = gr.update(value=None, visible=False)
    chips_reset = gr.update(samples=[], visible=False)
    yield history, gallery_reset, chips_reset
    
    full_response = ""
    products: List[Dict[str, Any]] = []
    options: List[Any] = []
    
    # Process the first chunk if it wasn't an ACK (or if we fetched a new one after ACK)
    # We need to handle it exactly like the loop handles chunks
    current_chunk = first_chunk
    
    # Helper to process a chunk
    def process_chunk(c):
        nonlocal full_response, products, options
        c_type, payload = _parse_stream_chunk(c)
        if c_type == "products":
            if isinstance(payload, list):
                products = payload
                agent_logger.info("UI received products", count=len(products))
            return False # Not text
        if c_type == "options":
            if isinstance(payload, list):
                options = payload
                agent_logger.info("UI received options", count=len(options))
            return False # Not text
        
        full_response += c
        return True # Is text

    if current_chunk:
        if process_chunk(current_chunk):
            history[-1]["content"] = full_response
            yield history, None, None

    async for chunk in generator:
        if process_chunk(chunk):
            history[-1]["content"] = full_response
            yield history, None, None

    gallery_update = gr.update(value=None, visible=False)
    if products:
        gallery_update = gr.update(
            value=_format_gallery_data(products),
            visible=True,
        )

    chips_update = gr.update(samples=[], visible=False)
    if options:
        chips_update = gr.update(
            samples=_format_option_samples(options),
            visible=True,
        )

    yield history, gallery_update, chips_update

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
