# app/gradio.py

import gradio as gr
from app.responses_agent import run_conversation  # or app.production_agent if that's the filename


def build_context_from_history(history, user_msg: str) -> str:
    """
    history is a list of OpenAI-style message dicts:
    [{"role": "user"|"assistant"|"system", "content": "..."}]

    We compress the last few turns into a short textual context
    so follow-ups like "Any other suggestions?" still refer to
    the previous fashion topic.
    """
    if not history:
        return user_msg.strip()

    ctx_lines = ["Previous conversation (last few turns):"]

    # Use last ~6 messages to avoid huge prompts
    for msg in history[-6:]:
        role = msg.get("role")
        content = msg.get("content", "")
        if not content:
            continue
        if role == "user":
            ctx_lines.append(f"User: {content}")
        elif role == "assistant":
            ctx_lines.append(f"MuseBot: {content}")

    ctx_lines.append("")
    ctx_lines.append(f"New user message: {user_msg.strip()}")
    ctx_lines.append(
        "Respond based on this new message and the previous context above."
    )

    return "\n".join(ctx_lines)


async def chat_fn(message, history, user_id):
    """
    Gradio ChatInterface callback.

    - message: latest user message (string)
    - history: list of messages (OpenAI style dicts) so far
    - user_id: from the textbox
    """
    if not message or not message.strip():
        return "Please ask me something fashion-related. 👕"

    effective_msg = build_context_from_history(history, message)

    # Thread id can just be per-user; backend is single-turn anyway
    thread_id = f"webui-{user_id}"

    reply = await run_conversation(
        user_id=user_id,
        message=effective_msg,
        thread_id=thread_id,
    )
    return reply


demo = gr.ChatInterface(
    fn=chat_fn,
    type="messages",  # OpenAI style: list of dicts with role/content
    title="🎨 MuseBot — Fashion Assistant",
    description="Ask me for outfits, pieces, or styling tips.",
    additional_inputs=[
        gr.Textbox(label="User ID", value="demo_user"),
    ],
    # IMPORTANT: with additional_inputs, examples must be list-of-lists:
    # [ <chat message>, <user_id> ]
    examples=[
        ["Hey! I'm looking for a blue cotton shirt", "demo_user"],
        ["What goes well with black jeans?", "demo_user"],
        ["Show me some summer outfits", "demo_user"],
        ["I want striped shirts for a beach vacation", "demo_user"],
    ],
    # No retry_btn / undo_btn / clear_btn for this Gradio version
)


if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )
