**Coding an LLM Architecture**
-
    -LLMs such as generative pretrained transformer are large deep neural network architecture desinged to generate new text one word at a time.
    - A GPT model in addition to the embedding layers, It consists of one or more transformer blocks containing the masked multi-head attention.
    -  ``` 
            GPT_CONFIG_124M = {
            "vocab_size": 50257,     # Vocabulary size
            "context_length": 1024,  # Context length
            "emb_dim": 768,          # Embedding dimension
            "n_heads": 12,           # Number of attention heads
            "n_layers": 12,          # Number of layers
            "drop_rate": 0.1,        # Dropout rate
            "qkv_bias": False        # Query-Key-Value bias
            }  
      ```