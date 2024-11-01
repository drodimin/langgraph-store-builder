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
    
    def __str__(self):
        return f"\033[92m{self.metadata}\033[0m\n{self.text}"
