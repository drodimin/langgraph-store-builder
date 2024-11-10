```mermaid
flowchart TD
    %% Define styles
    classDef default fill:#4285f4,stroke:#4285f4,color:white
    classDef startNode fill:#34a853,stroke:#34a853,color:white,rx:15
    classDef endNode fill:#ea4335,stroke:#ea4335,color:white,rx:15
    classDef desc fill:#ffffff,stroke:#cccccc,color:black,font-size:12px
    classDef subgraphnode fill:#4285f4,stroke:#4285f4,color:white,font-size:20px

    START[START]:::startNode
    
    subgraph Split ["Split Text"]
        Split-DESC["Splits text into independent chunks<br>preserving complete ideas and concepts"]
    end
    
    subgraph Adjust ["Adjust Chunks"]
        Adjust-DESC["Identifies any context lost upon<br>separation and adds it to make chunks<br>independently understandable"]
    end
    
    subgraph Iterate ["Iterate Chunks"]
        Iterate-DESC["Process each chunk sequentially.<br>Search vector store for existing entries<br>with high level of similarity"]
    end

    subgraph Prompt ["Human In The Loop"]
        Prompt-DESC["Let user evaluate current chunk against similar existing content and decide wether to store or skip"]
    end

    subgraph Store ["Store Chunk"]
        Store-DESC["Save chunk to vector database"]
    end

    subgraph Check ["Check End"]
        Check-DESC["Either returns to Iterate<br>or completes processing"]
    end

    END[END]:::endNode

    %% Main flow with vertical orientation
    START --> Split --> Adjust --> Iterate
    Iterate --> |"Similar chunk found"| Prompt
    Iterate --> |"No similar chunk"| Store
    Prompt --> |"Store"| Store
    Prompt --> |"Skip"| Check
    Store --> Check
    Check --> |"More chunks"| Iterate
    Check --> |"No more chunks"| END

    %% Style applications
    class Split-DESC,Adjust-DESC,Iterate-DESC,Prompt-DESC,Store-DESC,Check-DESC desc
    class Split,Adjust,Iterate,Prompt,Store,Check subgraphnode
