from chunk import Chunk
from chunk_store import ChunkPineconeStore
from configuration import Configuration
from graph_state import GraphState
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import adispatch_custom_event
from langgraph.errors import NodeInterrupt
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os

load_dotenv()

pc = Pinecone()
pc_index_name = os.getenv("PINECONE_INDEX_NAME")

if pc_index_name not in pc.list_indexes().names():
    pc.create_index(
        name=pc_index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Index '{pc_index_name}' created successfully.")
else:
    print(f"Index '{pc_index_name}' already exists.")

index = pc.Index(pc_index_name)

embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"
)

sonnet = ChatAnthropic(model='claude-3-5-sonnet-20241022', temperature=0)
openai = ChatOpenAI(model="gpt-4o-mini", temperature=0)

indexes = ChunkPineconeStore(vectorstore)

prompt = """You are a text processing assistant. Your task is to break a given text into self-contained chunks if possible. Here's the text you'll be working with:

<text>{text}</text>

Please follow these steps to process the text:

1. Read and analyze the entire text carefully.

2. Identify natural break points where the text can be divided into self-contained chunks following these guidelines:
- Each chunk should represent a complete thought or idea. 
- Keep together sentences that complement each other such one sentence makes statement and the next one provides an example.
- It is ok not to break the text into chunks if it is already self-contained in which case you should return the entire text as a single chunk.

3. Extract these chunks from the text, maintaining the original wording and punctuation within each chunk.

4. Present each chunk on a new line, with an empty line separating each chunk. Do not use any other formatting such as identation, numbering, etc.

5. Do not include any additional output besides resulting chunks of text

Now, please proceed with processing the input text according to these instructions."""

split_prompt = ChatPromptTemplate.from_messages([
    ("system", ""),
    ("human", prompt)
])

prompt = """You will be given a short chunk of text. Your task is to extract keywords from this text. Here is the text:

<text>
{text}
</text>

To extract keywords, follow these guidelines:
1. Look for names or titles representing technologies, companies, products, or other entities.
2. If there are no clear names or titles, identify unique terms that represent specific areas of work, industries, or concepts.
3. Avoid verbs, common adjectives, and general nouns.
4. Focus on nouns and proper nouns that are central to the main idea of the text.
5. Typically, you should aim to extract 1-5 keywords, depending on the length and complexity of the text.

Here are some examples to guide you:

Good keyword extraction:
Text: "Apple unveiled its latest iPhone model with advanced AI capabilities."
Keywords: Apple, iPhone, AI

Bad keyword extraction:
Text: "The company released a new product last week."
Keywords: company, released, product, week

Present your extracted keywords in a comma-separated list. Do not include any additional output besides the keywords.

Think through your keyword selection process carefully before providing your final answer. Consider the relevance and specificity of each potential keyword.

Now, please extract the keywords from the given text and present them as instructed."""

keyword_prompt = ChatPromptTemplate.from_messages([
    ("system", ""),
    ("human", prompt)
])

async def split_text(state: GraphState, config: RunnableConfig):
    chain = split_prompt | sonnet
    if not state["text"] or state["text"].strip() == "Your input text here":
        raise ValueError("Please provide actual text content to process")
    response = chain.invoke({"text": state["text"]})

    chunks = []
    for i, chunk in enumerate(response.content.split("\n\n")):
        chunk_text = chunk.strip()
        if not chunk_text:
            continue
        
        print('\nChunk #', i, '\033[93m', chunk_text, '\033[0m')
        new_chunk = Chunk("", chunk_text)
        chunks.append(new_chunk)

    print('Text split into', len(chunks), 'chunks\n')

    # Send the full Chunk objects in the event
    await adispatch_custom_event(
        "on_text_split",
        {
            "chunk_count": len(chunks),
            "chunks": chunks  # Send complete Chunk objects
        },
        config=config
    )

    return {"chunks": chunks, "index": -1}

def get_chunk_from_state(state: GraphState, index: int) -> Chunk:
    chunk_data = state["chunks"][index]
    if isinstance(chunk_data, dict):
        return Chunk(text=chunk_data['text'], metadata=chunk_data['metadata'])
    return chunk_data

async def iterate_chunks(state: GraphState):
    currentIndex = state["index"] + 1
    print("\033[91mIterate " + str(currentIndex + 1) + "/" + str(len(state["chunks"])) + "\033[0m")
    currentChunk = get_chunk_from_state(state, currentIndex)
    print("Text:\033[92m", currentChunk.text, "\033[0m")
    
    # Get keywords for the current chunk
    chain = keyword_prompt | sonnet
    response = chain.invoke({"text": currentChunk.text})
    currentChunk.metadata = response.content
    print('Keywords:\033[92m', currentChunk.metadata, '\033[0m')

    # Update the chunk in the state with its new metadata
    state["chunks"][currentIndex] = currentChunk
    state["index"] = currentIndex
    
    # Find similar chunks
    similar_doc = find_similar_chunk(state)
    if similar_doc:
        print('\033[91mSimilar chunk found\033[0m')
    state["similar_chunk"] = Chunk(similar_doc.metadata['keywords'], similar_doc.page_content) if similar_doc else None

    # Dispatch event for UI update with new metadata
    await adispatch_custom_event(
        "on_chunk_metadata_update",
        {
            "chunk_index": currentIndex,
            "chunk": currentChunk
        },
        config=RunnableConfig()
    )

    return state

def find_similar_chunk(state: GraphState):
    chunk = get_chunk_from_state(state, state["index"])
    return indexes.findChunk(chunk)

def chunk_action(state: GraphState):
    if state["similar_chunk"] is None:
        return "store"
    else:
        return "prompt"

async def index_chunk(state: GraphState):
    print('\033[93mIndexing chunk\033[0m')
    state["answer"] = None
    chunk = get_chunk_from_state(state, state["index"])
    
    indexes.addChunk(chunk)

    await adispatch_custom_event(
        "on_chunk_result",
        {
            "chunk_index": state["index"],
            "result": "Indexed"
        },
        config=RunnableConfig()
    )

    return state

async def prompt_chunk(state: GraphState, config: RunnableConfig):
    answer = state.get("answer")
    print("\033[91mEntering prompt_chunk\033[0m, answer:", answer)
    similar_chunk = state["similar_chunk"]
    if isinstance(similar_chunk, dict):
        similar_chunk = Chunk(text=similar_chunk['text'], metadata=similar_chunk['metadata'])
    
    if answer is not None:
        return {
            **state,
            "answer": answer,
            "similar_chunk": None
        }
    
    current_chunk = get_chunk_from_state(state, state["index"])
    await adispatch_custom_event(
        "on_similar_chunk_found", 
        {
            "chunk_index": state["index"],
            "chunk_text": similar_chunk.text,
            "chunk_keywords": similar_chunk.metadata,
            "current_chunk": current_chunk  # Include the current chunk with its metadata
        }, 
        config=config
    )
    
    if config.get("is_graph_studio"):
        return {
            **state,
            "waiting_for_input": True,
            "options": ["y", "n"],
            "prompt_message": "Do you want to index this chunk?",
        }
    
    print("\033[93mRaising interrupt for user decision\033[0m")
    raise NodeInterrupt("Do you want to index this chunk?")

async def process_decision(state: GraphState):
    print("\033[93mEntering process_decision with state\033[0m")
    answer = state.get("answer")
    
    print(f"\033[93mProcessing decision with answer: {answer}\033[0m")

    if answer == "y":
        result = "store"
    else:
        result = "endcheck"
        await adispatch_custom_event(
            "on_chunk_result",
            {
                "chunk_index": state["index"],
                "result": "Skipped"
            },
            config=RunnableConfig()
        )

    
    print(f"\033[93mDecision result: {result}\033[0m")
    return result

def end_check(state: GraphState):
    state["answer"] = None
    return state

def is_end(state: GraphState):
    print("\033[91mIsEnd:" + str(state["index"] + 1) + "/" + str(len(state["chunks"])) + "\033[0m\n")
    if state["index"] >= len(state["chunks"])-1:
        return END
    else:
        return "iterate"

workflow = StateGraph(GraphState, config_schema=Configuration)
workflow.add_node("split", split_text)
workflow.add_node("iterate", iterate_chunks)
workflow.add_node("prompt", prompt_chunk)
workflow.add_node("store", index_chunk)
workflow.add_node("endcheck", end_check)

workflow.add_edge(START, "split")
workflow.add_edge("split", "iterate")
workflow.add_conditional_edges("iterate", chunk_action)
workflow.add_conditional_edges("prompt", process_decision)
workflow.add_edge("store", "endcheck")
workflow.add_conditional_edges(
    "endcheck", 
    is_end,
    {END: END, "iterate": "iterate"}
)

memory = MemorySaver()
print("memory saver is rerun")
app = workflow.compile(checkpointer=memory)

def create_app():
    return app

