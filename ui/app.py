import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/chat"


st.set_page_config(
    page_title="E-commerce Assistant",
    page_icon="🛍️"
)

st.title("🛍️ E-commerce Assistant")

st.caption(
    "Ask me about products, recommendations, comparisons, "
    "delivery, returns, refunds, and other policies."
)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if st.button("New Chat"):

    st.session_state.messages = []
    st.session_state.thread_id = None

    st.rerun()

# ---------------------------------------------------------
# DISPLAY EXISTING CHAT
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )


# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------

user_message = st.chat_input(
    "Ask a product or policy question..."
)


if user_message:

    # Save user message locally
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_message)


    # Build request for FastAPI
    payload = {
        "message": user_message,
        "thread_id": st.session_state.thread_id
    }


    try:

        with st.spinner("Thinking..."):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=300
            )

        response.raise_for_status()

        data = response.json()


        # Save the thread ID returned by FastAPI.
        # First request:
        # None -> FastAPI generates an ID.
        #
        # Later requests:
        # This same ID is sent back automatically.
        st.session_state.thread_id = data[
            "thread_id"
        ]

        st.sidebar.write(
        "Thread ID:",
        st.session_state.thread_id
        )
        assistant_answer = data["answer"]


        # Save assistant message
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_answer
            }
        )


        # Display assistant message
        with st.chat_message("assistant"):

            st.markdown(
                assistant_answer
            )


    except requests.exceptions.ConnectionError:

        st.error(
            "Unable to connect to the chatbot API. "
            "Make sure FastAPI is running."
        )


    except requests.exceptions.RequestException as exc:

        st.error(
            f"Chatbot request failed: {exc}"
        )