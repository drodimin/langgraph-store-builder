from chunk import Chunk
from chunk_store import ChunkPineconeStore
from configuration import Configuration
from graph_state import GraphState
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

pc = Pinecone()
pc_index_name = "langgraph-test"

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

prompt = """You are a text processing assistant. Your task is to break a given text into self-contained chunks, each representing a complete thought or idea. Here's the text you'll be working with:

<text>{text}</text>

Please follow these steps to process the text:

1. Read and analyze the entire text carefully.

2. Identify natural break points where the text can be divided into self-contained chunks. Each chunk should represent a complete thought or idea. Keep together sentences that complement each other such when an example follows a statement

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

def split_text(state: GraphState) -> GraphState:
    chain = split_prompt | sonnet
    response = chain.invoke({"text": state["text"]})

    chunks = []
    for chunk in response.content.split("\n\n"):
        chunk_text = chunk.strip()
        if not chunk_text:
            continue

        new_chunk = Chunk("", chunk_text)
        print("Created chunk type:", type(new_chunk))
        chunks.append(new_chunk)

    print('Text split into', len(chunks), 'chunks')
    return {"chunks": chunks, "index": -1}

def get_chunk_from_state(state: GraphState, index: int) -> Chunk:
    chunk_data = state["chunks"][index]
    if isinstance(chunk_data, dict):
        return Chunk(text=chunk_data['text'], metadata=chunk_data['metadata'])
    return chunk_data

def iterate_chunks(state: GraphState):
    currentIndex = state["index"] + 1
    print("Iterate " + str(currentIndex) + "/" + str(len(state["chunks"])))
    currentChunk = get_chunk_from_state(state, currentIndex)
    print("Iterate chunk type:", type(currentChunk))

    chain = keyword_prompt | sonnet
    response = chain.invoke({"text": currentChunk.text})
    currentChunk.metadata = response.content

    state["index"] = currentIndex
    similar_doc = find_similar_chunk(state)
    state["similar_chunk"] = Chunk(similar_doc.metadata['keywords'], similar_doc.page_content) if similar_doc else None

    return state

def find_similar_chunk(state: GraphState):
    chunk = get_chunk_from_state(state, state["index"])
    print("Finding similar chunk type:", type(chunk))
    return indexes.findChunk(chunk)

def chunk_action(state: GraphState):
    if state["similar_chunk"] is None:
        return "store"
    else:
        return "prompt"

def index_chunk(state: GraphState):
    chunk = get_chunk_from_state(state, state["index"])
    print("Index chunk type:", type(chunk))
    print("Index chunk content:", chunk)
    
    indexes.addChunk(chunk)
    return state

def prompt_chunk(state: GraphState, config: RunnableConfig):
    print("Decide:" + str(state["index"]))
    similar_chunk = state["similar_chunk"]
    if isinstance(similar_chunk, dict):
        similar_chunk = Chunk(text=similar_chunk['text'], metadata=similar_chunk['metadata'])
    print("Similar chunk has been indexed:" + str(similar_chunk))
    
    return {
        **state,
        "waiting_for_input": True,
        "options": ["y", "n"],
        "prompt_message": "Do you still want to index this chunk?",
        "answer": None
    }

def process_decision(state: GraphState):
    if state.get("answer") == "y":
        return "store"
    else:
        return "endcheck"

def end_check(state: GraphState):
    return state

def is_end(state: GraphState):
    print("IsEnd:" + str(state["index"]) + "/" + str(len(state["chunks"])))
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

app = workflow.compile()