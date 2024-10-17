**There are 4 different variants of attention mechanisms**
-
    - Simplified self attention
    - Self-attention
    - Casual attention
    - Multi-head attention


- Before LLMs to translate from one language to another, long contextual understanding is required. 
  - To address this problem encoder and decoder are the two submodules proposed. Encoder is to first read in and process entire text, decoder then produces the translated text
  - Before the advent of Transformers, RNNs were the most popular encoder-decoder architectures
  - The big limitation of encoder-decoder RNNs is loss of context
    - RNNs cant directly access earlier hidden states, it soley relies on current hidden state, which encapsulates all relevant information

**3.2 Capturing data dependencies with attention mechanisms**
-
- RNNs are good for shorter sentences, but not for longer texts. 
- Researchers developed the Bahdanau attention mechanism for RNNs in 2014
- This means some input tokes are more important than others for generating output token, this importance is due to attention weights
- 3 years later in 2017, original transformer architecture papers were released
- Self-attention mechanism is a mechanism that allows each position in the input sequence to consider relevancy of all other positions in same sequence.
- self-attention is the key component of contemporary LLMs, such as GPT series

**3.3 Attending different parts of the input with self-attention**

- "self" refers to the mechanism's ability to compute attention weights by relating different positions within a single sequence, such as word sentence or pixels in an image.
- In traditional attention mechanism, the focus is on the relationship between 2 different sequences, such as input sequence and an output sequence.

**3.3.1 A simple self attention mechanism without trainable weights**

- An input sequence like "your journey starts with one step", x is an input sequence. 
- The goal here is to calculate context vectors z(i) for each element x(i)
- context vectors play a crucial role in self-attention. 
  - Their purpose is to create an enriched representation of each element in an input sequence.
- Unlike in self-attention mechanism, later by adding trainable weights, LLM will learn to construct these context vectors which will help LLMs to generate the next token.
- The first step in implementing self-attention is to calculate the intermediate values also called as attention scores.
- Beyond viewing dot product as mathematical operation, it is a measure of similarity because, it quantifies how close those two vectors are aligned.
  - Higher the dot product, it indicates greater degree of similarity between the vectors
  - Lower the dot product, they are far apart from each other
- Next step is to normalize these attention scores, to main the training Stability
- Attention weights are derived from attention score, its the normalized value which will sum's upto 1
  - Normalization is better at managing extreme values and offers more favourable gradient properties during training.
  - Softmax function ensures that attention weights are always positive
- 