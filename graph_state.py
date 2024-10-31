from typing import TypedDict, Optional, List

from chunk import Chunk
class GraphState(TypedDict, total=False):
  text: str
  chunks: list[Chunk]
  index: int
  similar_chunk: Optional[Chunk]
  prompt_message: str
  answer: Optional[str]