import streamlit as st
import os
import re
import base64

from chat import Chat, load_all_chats
from answer import answer_question_stream
from datetime import datetime

from utils import (
    count_tokens,
    split_text_into_chunks,
    new_id,
    now,
    MAX_TOKENS
)

# ---------------- paths ----------------
CSS_PATH = 'css/styles.css'
LOGO_PATH = "logo/hooshdaroo.png"

CHAT_DIR = "chats"
os.makedirs(CHAT_DIR, exist_ok=True)


# ---------------- streamlit config ----------------
def setup_config():
    config_dir = ".streamlit"
    config_file = os.path.join(config_dir, "config.toml")

    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            f.write("""[theme]
primaryColor = "#6b6d6f"
backgroundColor = "#1e1f23"
secondaryBackgroundColor = "#24252b"
textColor = "#9b9d9d"
font = "sans serif"

[server]
headless = false

[browser]
gatherUsageStats = false

[client]
toolbarMode = "minimal"
""")


setup_config()


# ---------------- image to base64 ----------------
def get_base64_image(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


# ---------------- page config ----------------
st.set_page_config(
    page_title="Hoosh Daroo",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------- Load External CSS ----------------
def load_css(file_path: str) -> None:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            css = f.read()
            st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found: {file_path}")


load_css(CSS_PATH)


# ---------------- text helpers ----------------
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def is_persian(text: str) -> bool:
    return re.search(r'[\u0600-\u06FF]', text or "") is not None


def sanitize_for_display(text: str) -> str:
    """Remove HTML tags that sometimes leak into outputs (e.g., <div>...</div>)."""
    if not text:
        return ""
    # Remove all HTML tags to prevent them showing up as literal 'div' boxes
    text = _HTML_TAG_RE.sub('', text)
    return text


def render_text(text: str) -> None:
    text = sanitize_for_display(text)
    direction = "rtl" if is_persian(text) else "ltr"
    align = "right" if direction == "rtl" else "left"

    # NOTE: we keep unsafe_allow_html=True only for direction/alignment.
    # We sanitize HTML tags above to prevent stray <div> from appearing.
    st.markdown(
        f"""
        <div dir="{direction}" style="text-align:{align};">
        {text}
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------- session state ----------------
if "chats" not in st.session_state:
    loaded = load_all_chats()
    st.session_state.chats = {
        cid: {
            "chat": chat,
            "title": chat.title,
            "creation_time": datetime.fromtimestamp(os.path.getctime(chat.file_path)),
        }
        for cid, chat in loaded.items()
    }

if "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.chats:
    new_chat = Chat()
    chat_id = new_chat.chat_id
    st.session_state.chats[chat_id] = {
        "chat": new_chat,
        "title": "New chat",
        "creation_time": datetime.now(),
    }
    st.session_state.current_chat_id = chat_id


# ---------------- sidebar ----------------
with st.sidebar:
    st.logo(LOGO_PATH, size="large")
    st.markdown('<h1><span class="material-icons">chat_bubble</span> Chats</h1>', unsafe_allow_html=True)

    if st.button(":material/Add_Box: New Chat", type='tertiary', use_container_width=True):
        new_chat = Chat()
        chat_id = new_chat.chat_id
        st.session_state.chats[chat_id] = {
            "chat": new_chat,
            "title": "New chat",
            "creation_time": datetime.now(),
        }
        st.session_state.current_chat_id = chat_id
        if "editing_chat_id" in st.session_state:
            del st.session_state.editing_chat_id
        st.rerun()

    sorted_chats = sorted(
        st.session_state.chats.items(),
        key=lambda x: x[1]["creation_time"],
        reverse=True,
    )

    for chat_id, info in sorted_chats:
        editing = st.session_state.get("editing_chat_id") == chat_id
        if editing:
            new_title = st.text_input(
                "Rename chat",
                value=info["title"],
                key=f"rename_input_{chat_id}",
            )
            if st.button("Save", help='save', key=f"save_{chat_id}"):
                if new_title.strip():
                    info["chat"].set_title(new_title.strip())
                    st.session_state.chats[chat_id]["title"] = new_title.strip()
                del st.session_state.editing_chat_id
                st.rerun()
        else:
            col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
            is_current = (chat_id == st.session_state.current_chat_id)
            button_type = "primary" if is_current else "tertiary"

            with col1:
                if st.button(info["title"], help="switch chat", type=button_type, key=f"switch_{chat_id}"):
                    st.session_state.current_chat_id = chat_id
                    if "editing_chat_id" in st.session_state:
                        del st.session_state.editing_chat_id
                    st.rerun()
            with col2:
                if st.button(":material/edit: ", help='edit chat name', type='tertiary', key=f"edit_{chat_id}"):
                    st.session_state.editing_chat_id = chat_id
                    st.rerun()
            with col3:
                if st.button(":material/delete: ", help='delete', type='tertiary', key=f"delete_{chat_id}"):
                    try:
                        os.remove(info["chat"].file_path)
                    except Exception:
                        pass
                    del st.session_state.chats[chat_id]
                    if st.session_state.current_chat_id == chat_id:
                        if st.session_state.chats:
                            st.session_state.current_chat_id = next(iter(st.session_state.chats))
                        else:
                            new_chat = Chat()
                            new_id_ = new_chat.chat_id
                            st.session_state.chats[new_id_] = {
                                "chat": new_chat,
                                "title": "New chat",
                                "creation_time": datetime.now(),
                            }
                            st.session_state.current_chat_id = new_id_
                    if "editing_chat_id" in st.session_state:
                        del st.session_state.editing_chat_id
                    st.rerun()


# ---------------- main ----------------
current_chat_id = st.session_state.current_chat_id
chat = st.session_state.chats[current_chat_id]["chat"]


try:
    logo_base64 = get_base64_image(LOGO_PATH)

    st.markdown(
        f'''
        <div style="width: 100%; text-align: center; direction: rtl; padding: 1rem 0;">
            <div style="display: inline-flex; align-items: center; gap: 1rem; flex-direction: row-reverse;">
                <span style="font-family: 'Noto Nastaliq Urdu', serif; font-size: 3rem; font-weight: 700;">
                    هوش دارو
                </span>
                <img src="data:image/png;base64,{logo_base64}" style="height: 8rem; width: auto;">
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        '''
        <h1 style="direction: rtl; text-align: center; font-family: 'Vazirmatn', sans-serif; font-size: 1.5rem; font-weight: 200; border: none; padding: 2rem 0 1rem 0;">
            دستیار داروساز هوشمند
        </h1>
        ''',
        unsafe_allow_html=True
    )

except Exception:
    st.markdown(
        '''
        <h1 style="direction: rtl; text-align: center; font-family: 'Noto Nastaliq Urdu', serif;">
            هوش دارو
        </h1>
        ''',
        unsafe_allow_html=True
    )


# ---------------- render messages ----------------
for msg in chat.messages:
    with st.chat_message(msg["role"]):
        render_text(msg.get("content", ""))
        if msg["role"] == "assistant" and msg.get("references"):
            with st.expander("References"):
                for ref in msg["references"]:
                    source = sanitize_for_display(str(ref.get("source", "unknown")))
                    sections = sanitize_for_display(str(ref.get("sections", "")))
                    name = sanitize_for_display(str(ref.get("name", "")))
                    st.markdown(
                        f"""
                        **Name:** {name}  
                        **Source:** {source}  
                        **Sections:** `{sections}`  
                        ---
                        """
                    )


# ---------------- input ----------------
prompt = st.chat_input("Ask something...")

if prompt:
    prompt = sanitize_for_display(prompt)

    with st.chat_message("user"):
        render_text(prompt)
    chat.add_message("user", prompt)

    history = chat.context_for_llm()
    generator, retrieved_chunks= answer_question_stream(prompt, history)

    full_answer = ""

    with st.chat_message("assistant"):
        placeholder = st.empty()

        for token in generator:
            full_answer += token
            # IMPORTANT: prevent stray HTML tags like <div> from showing up.
            placeholder.markdown(f"""
                            <div dir="rtl" style="text-align:right;">{sanitize_for_display(full_answer)}</div>""",
                            unsafe_allow_html=True
                                )

    full_answer = sanitize_for_display(full_answer)
    chat.add_message(
        role="assistant",
        content=full_answer,
        references=retrieved_chunks
    )

    st.rerun()
