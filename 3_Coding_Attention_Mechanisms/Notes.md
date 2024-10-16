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
    - RNNs are good for shorter sentences, but not for longer texts. 
    - Researchers developed the Bahdanau attention mechanism for RNNs in 2014
    - This means some input tokes are more important than others for generating output token, this importance is due to attention weights
    - 3 years later in 2017, original transformer architecture papers were released
    - Self-attention mechanism is a mechanism that allows each position in the input sequence to consider relevancy of all other positions in same sequence.
    - self-attention is the key component of contemporary LLMs, such as GPT series
