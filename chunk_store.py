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
        self.similarity_threshold = 0.85

    def addChunk(self, chunk):
        document = Document(
            page_content=chunk.text,
            metadata={"keywords": chunk.metadata}
        )
        stored_id = self.vectorstore.add_documents([document])[0]
        return stored_id

    def findChunk(self, chunk):
        results = self.vectorstore.similarity_search_with_score(chunk.text, k=5)
        if len(results) == 0:
            return None

        for doc, score in results:
            similarity = 1 - score
            metadata_match = doc.metadata.get('keywords') == chunk.metadata
            if similarity >= self.similarity_threshold and metadata_match:
                return doc

        return None
  
