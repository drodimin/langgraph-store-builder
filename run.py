from graph import create_app
from langchain_core.runnables import RunnableConfig

def run_graph(input_text: str):
    app = create_app()
    config = RunnableConfig(is_graph_studio=False)
    # Create initial state with the input text
    initial_state = {
        "text": input_text,
        "chunks": [],
        "index": 0,
        "similar_chunk": None
    }
    return app.invoke(initial_state, config=config)

if __name__ == "__main__":
    input_text = input("Please enter your text: ")
    output = run_graph(input_text)
    print("\nOutput:", output)

