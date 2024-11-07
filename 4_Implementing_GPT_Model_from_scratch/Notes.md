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

**4.2 Normalizing activations with layer normalization**
-
    - Training deep neural networks with many layers is challenging due to vanishing or exploding gradients. which means, the network has difficutly learning the underlying patterns in the data to a degree that would allow it to make predictions or decisions.
    - The main idea behind layer normalization is to adjust the activations of a neural network layer to have a mean of 0 and a variance of 1.
        - This adjustment speeds up the convergence to effective weights and ensures consistent, reliable training.
    - Layer normalization is typically applied before and after the multi-head attention module.

**4.3 Implementing a feed forward network with GELU activations**
-
    - GELU activation function plays a crucial role in the neural network submodule.
    
**short connections**
-
    - Shortcut connections are important for overcoming the limitations posed by the vanishing gradient problem in deep neural networks. 


