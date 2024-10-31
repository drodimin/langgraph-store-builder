from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    metadata: str

    def __init__(self, metadata: str = "", text: str = ""):
        self.text = text
        self.metadata = metadata

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "metadata": self.metadata
        }