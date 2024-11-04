# app.py
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
import asyncio
from graph import app as graph
from langgraph.errors import NodeInterrupt

DEBUG_MODE = False

def debug_print(container, message):
    if DEBUG_MODE:
        container.write(f"Debug: {message}")

thread_config = {"configurable": {"thread_id": "1"}}

def display_text_chunks(chunks):
    """Display text chunks in an expander with consistent formatting"""
    with st.expander(f"Text is split into {len(chunks)} chunks:", expanded=True):
        for i, chunk in enumerate(chunks, 1):
            st.markdown(f"**Chunk {i}:**")
            st.text_area(
                label=f"Chunk {i}",
                value=chunk,
                height=100,
                label_visibility="collapsed"
            )
            st.divider()

async def process_graph_events(text_input, placeholder, chunks_container, shared_state):
    """Handle graph events and UI updates"""
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
            input_data = None
        else:
            input_data = {"text": text_input}
        
        async for event in graph.astream_events(input_data, config, version="v2"):
            name = event["name"]
            debug_print(container, f"Received event: {name}")
            
            if name == "on_text_split":
                st.session_state.text_chunks = event['data']['chunks']
                st.session_state.chunks_received = True
                
                with chunks_container:
                    display_text_chunks(st.session_state.text_chunks)
            
            elif name == "on_similar_chunk_found":
                chunk_num = event['data']['chunk_index'] + 1  # Convert to 1-based index
                container.warning(f"Found existing item similat to Chunk #{chunk_num}:")
                
                container.subheader(f"Existing Item:")
                container.info(f"Keywords: {event['data']['chunk_keywords']}")
                container.code(event['data']['chunk_text'], wrap_lines=True)
                
                container.write("Decision required:")
                col1, col2 = st.columns(2)
                store = col1.button("Store Chunk", key=f"store_{chunk_num}")
                skip = col2.button("Skip Chunk", key=f"skip_{chunk_num}")
                
                if store or skip:
                    answer = "y" if store else "n"
                    debug_print(container, f"Button clicked, answer: {answer}")
                    return {"status": "resume", "answer": answer}
                return {"status": "waiting", "message": "Do you want to index this chunk?"}
    
        return {"status": "completed"}
        
    except NodeInterrupt as e:
        debug_print(container, f"NodeInterrupt caught with message: {str(e)}")
        
        container.write("Decision required:")
        col1, col2 = st.columns(2)
        store = col1.button("Store Chunk", key=f"store_interrupt_{chunk_num}")
        skip = col2.button("Skip Chunk", key=f"skip_interrupt_{chunk_num}")
        
        if store or skip:
            answer = "y" if store else "n"
            debug_print(container, f"Button clicked in interrupt handler, answer: {answer}")
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
if "chunks_received" not in st.session_state:
    st.session_state.chunks_received = False

# Create main layout containers
form_container = st.container()
chunks_container = st.container()
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
            st.session_state.chunks_received = False  # Reset chunks flag on new submission

# Always show chunks container
with chunks_container:
    if st.session_state.text_chunks and st.session_state.chunks_received:
        display_text_chunks(st.session_state.text_chunks)

# Process text and handle decisions
if submit_button or st.session_state.graph_resume:
    with process_container:
        with st.spinner("Processing..."):
            shared_state = {
                "graph_resume": st.session_state.graph_resume,
                "user_answer": st.session_state.user_answer
            }
            
            result = asyncio.run(process_graph_events(
                st.session_state.input_text or text_input,
                process_container,
                chunks_container,
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
                st.session_state.chunks_received = False
                st.success("Text processing completed!")