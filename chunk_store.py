from chunk import Chunk
from langchain_core.documents import Document

class ChunkLocalStore:
  def __init__(self):
    self.chunks: list[Chunk] = []

  def addChunk(self, chunk: Chunk):
    self.chunks.append(chunk)

  def findChunk(self, chunk: Chunk):
    for c in self.chunks:
      if c.metadata == chunk.metadata and c.text == chunk.text:
        return c
    return None

  def __str__(self):
    chunk_strs = [str(chunk) for chunk in self.chunks]
    return "\n".join(chunk_strs)
  
class ChunkPineconeStore:
  def __init__(self, vectorstore):
    self.vectorstore = vectorstore
    self.similarity_threshold = 0.90  # Adjust this threshold as needed (0-1)

  def addChunk(self, chunk):
    document = Document(
        page_content=chunk.text,
        metadata={"keywords": chunk.metadata}
    )
    stored_id = self.vectorstore.add_documents([document])[0]
    return stored_id

  def findChunk(self, chunk):
    # Get more results to filter
    results = self.vectorstore.similarity_search_with_score(
        chunk.text,
        k=3  # Get top 5 results to filter
    )
    
    if len(results) == 0:
        return None

    # Filter results based on similarity score and metadata
    for doc, score in results:
        print("\033[93m- Checking similar doc with score", str(score))
        #print('\033[0m' + doc.page_content + '\033[0m')
        if score >= self.similarity_threshold:
            return doc

    return None
  
