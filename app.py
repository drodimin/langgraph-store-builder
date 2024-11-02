# app.py
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
import asyncio
from graph import app as graph
from langgraph.errors import NodeInterrupt

DEBUG_MODE = True

def debug_print(container, message):
    if DEBUG_MODE:
        container.write(f"Debug: {message}")

thread_config = {"configurable": {"thread_id": "1"}}

def display_text_chunks(chunks, container, prefix=""):
    """Display text chunks in an expander with consistent formatting"""
    # Clear the container before updating
    container.empty()

    with container:
        with st.expander(f"Text is split into {len(chunks)} chunks:", expanded=True):
            for i, chunk in enumerate(chunks, 1):
                st.markdown(f"**Chunk {i}:**")
                if hasattr(chunk, 'metadata') and chunk.metadata:
                    st.info(f"Keywords: {chunk.metadata}")
                chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                st.text_area(
                    key=f"{prefix}chunk_{i}_{id(chunk)}",
                    label=f"Chunk {i}",
                    value=chunk_text,
                    height=100,
                    label_visibility="collapsed"
                )
                if hasattr(chunk, 'result'):
                    st.success(f"Result: {chunk.result}")
                st.divider()

async def process_graph_events(text_input, placeholder, shared_state):
    """Handle graph events and update session state"""
    container = placeholder

    try:
        debug_print(container, "Starting graph processing...")

        config = {
            "configurable": {"thread_id": "1"}
        }

        if shared_state.get("graph_resume"):
            answer = shared_state.get("user_answer")
            debug_print(container, f"Resuming with answer: {answer}")
            graph.update_state(config, {"answer": answer})
            input_data = {"answer": answer}
        else:
            input_data = {"text": text_input}
            debug_print(container, "Starting new processing")

        async for event in graph.astream_events(input_data, config, version="v2"):
            name = event["name"]
            debug_print(container, f"Event received: {name}")

            if name == "on_text_split":
                st.session_state.text_chunks = event['data']['chunks']
                st.session_state.chunks_updated = True  # Flag to indicate chunks have been updated
                debug_print(container, "Chunks updated in session state")

            elif name == "on_chunk_metadata_update":
                chunk_index = event['data']['chunk_index']
                chunk = event['data']['chunk']
                if hasattr(st.session_state, 'text_chunks'):
                    st.session_state.text_chunks[chunk_index] = chunk
                    st.session_state.chunks_updated = True
                debug_print(container, f"Updated metadata for chunk {chunk_index}")

            elif name == "on_chunk_result":
                chunk_index = event['data']['chunk_index']
                if hasattr(st.session_state, 'text_chunks'):
                    setattr(st.session_state.text_chunks[chunk_index], 'result', event['data']['result'])
                    st.session_state.chunks_updated = True
                debug_print(container, f"Updated result for chunk {chunk_index}")

            elif name == "on_similar_chunk_found":
                # Handle decision logic here
                container.empty()
                chunk_num = event['data']['chunk_index'] + 1 

                container.warning(f"Found similar existing item for Chunk #{chunk_num}")

                container.subheader("Existing Item:")
                container.info(f"Keywords: {event['data']['chunk_keywords']}")
                container.code(event['data']['chunk_text'])

                container.subheader("Current Chunk:")
                container.info(f"Keywords: {event['data']['current_chunk'].metadata}")
                container.code(event['data']['current_chunk'].text)

                st.write("Decision required:")
                col1, col2 = st.columns(2)
                store = col1.button("Store Chunk")
                skip = col2.button("Skip Chunk")

                if store or skip:
                    answer = "y" if store else "n"
                    debug_print(container, f"Decision made: {answer}")
                    return {"status": "resume", "answer": answer}
                return {"status": "waiting", "message": "Do you want to index this chunk?"}

        return {"status": "completed"}

    except NodeInterrupt as e:
        debug_print(container, f"NodeInterrupt: {str(e)}")

        container.write("Decision required:")
        col1, col2 = st.columns(2)
        store = col1.button("Store Chunk")
        skip = col2.button("Skip Chunk")

        if store or skip:
            answer = "y" if store else "n"
            debug_print(container, f"Decision made: {answer}")
            return {"status": "resume", "answer": answer}

        return {"status": "waiting", "message": str(e)}

# Main Streamlit app
st.title("Text Processing with LangGraph")

# Initialize session state
if "graph_resume" not in st.session_state:
    st.session_state.graph_resume = False
if "user_answer" not in st.session_state:
    st.session_state.user_answer = None
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = None
if "input_text" not in st.session_state:
    st.session_state.input_text = None
if "chunks_updated" not in st.session_state:
    st.session_state.chunks_updated = False

# Create main layout containers
form_container = st.container()
chunks_container = st.container()  # Single instance of chunks container
process_container = st.container()

# Input form
with form_container:
    with st.form(key="text_input_form"):
        text_input = st.text_area(
            "Enter text to process",
            height=200,
            placeholder="Enter your text here..."
        )
        submit_button = st.form_submit_button("Process Text")
        if submit_button:
            st.session_state.input_text = text_input
            st.session_state.graph_resume = False  # Reset resume flag on new submission
            st.session_state.user_answer = None  # Reset user answer
            st.session_state.chunks_updated = False  # Reset chunks updated flag

# Process text and handle decisions
if (submit_button and st.session_state.input_text) or st.session_state.graph_resume:
    with process_container:
        with st.spinner("Processing..."):
            shared_state = {
                "graph_resume": st.session_state.graph_resume,
                "user_answer": st.session_state.user_answer
            }

            result = asyncio.run(process_graph_events(
                st.session_state.input_text,
                process_container,
                shared_state
            ))

            if result["status"] == "waiting":
                st.session_state.graph_resume = True
                st.info(result["message"])
            elif result["status"] == "resume":
                st.session_state.user_answer = result["answer"]
                st.session_state.graph_resume = True
                st.rerun()
            elif result["status"] == "completed":
                st.session_state.graph_resume = False
                st.session_state.user_answer = None
                st.success("Text processing completed!")

# Display chunks if they have been updated
if st.session_state.chunks_updated and st.session_state.text_chunks:
    display_text_chunks(st.session_state.text_chunks, chunks_container)
    st.session_state.chunks_updated = False  # Reset the flag
