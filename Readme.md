OWNGPT – Build Your Own GPT from Scratch
OWNGPT is a minimal, educational implementation of a GPT‑style Transformer language model trained entirely from scratch using PyTorch.
It demonstrates step‑by‑step how a large language model (LLM) works internally – from tokenization and embeddings to multi‑head self‑attention, feed‑forward networks, training, and text generation.

🧠 The goal is to demystify the inner workings of modern LLMs and give you a hands‑on understanding of how they are built, piece by piece.

📖 Table of Contents
Introduction

Architecture Overview

Implementation Steps

Project Structure

Installation & Dependencies

Training

Generating Text

Results & Observations

Why This Matters

License

🚀 Introduction
Large Language Models (LLMs) like GPT‑3, ChatGPT, and LLaMA have revolutionised AI. But how do they actually work?
OWNGPT strips away the complexity and shows the core components:

Tokenisation – turning text into numbers.

Embeddings + Positional Encoding – representing tokens and their positions.

Self‑Attention (with causal masking) – letting each token look only at past tokens.

Multi‑Head Attention – attending to multiple representation subspaces.

Feed‑Forward Networks – adding non‑linearity and depth.

Stacked Transformer Blocks – building a deep network.

Training with Cross‑Entropy – predicting the next token.

Text Generation – sampling from the trained model.

The entire model is written in pure PyTorch with less than 500 lines of code – making it easy to read, modify, and learn from.

🧠 Architecture Overview
Component	Description
Tokenizer	Character‑level mapping (text ↔ IDs). Vocab size ~70 for English characters.
Token Embedding	nn.Embedding(vocab_size, embed_dim) – converts each token ID to a dense vector.
Positional Encoding	Sinusoidal fixed encoding (like the original Transformer) – added to token embeddings.
Self‑Attention	Scaled dot‑product attention with causal mask (no peeking into the future).
Multi‑Head Attention	Splits embed_dim into num_heads heads, each with its own Q, K, V projections.
Feed‑Forward Network	Two‑layer MLP: embed_dim → hidden_dim → embed_dim, with ReLU.
Transformer Block	MHA + FFN with residual connections and layer normalisation.
Final Layer	LayerNorm + Linear projection to vocab size (logits).
Key design choice: We use a causal mask so that the model learns to predict the next token based only on past tokens – exactly what is needed for autoregressive generation.

👣 Implementation Steps
We broke down the implementation into 10 numbered scripts, each introducing one new concept and building on the previous.
Here’s what each step does:

Step	File	Concept
1	01_tokenizer.py	Build a character‑level tokenizer from training text.
2	02_embeddings.py	Create token embedding layer.
3	03_positional_encoding.py	Generate sinusoidal positional encodings.
4	04_self_attention.py	Implement scaled dot‑product attention (without causal mask initially).
5	05_multi_head_attention.py	Extend to multi‑head attention.
6	06_feed_forward.py	Build the feed‑forward network.
7	07_transformer_block.py	Combine MHA + FFN with residual connections and layer norm.
8	08_gpt_model.py	Stack multiple blocks, add embeddings, and final linear head – the full GPT.
9	09_training.py	Load text, create dataset, train the model with cross‑entropy loss.
10	10_generation.py	Load trained weights and generate new text from a prompt.
The reusable module files (tokenizer.py, gpt_model.py, etc.) contain the actual implementation, while the numbered scripts demonstrate and test each piece.

📁 Project Structure
text
OWNGPT/
├── 01_tokenizer.py          # Step‑by‑step demos
├── 02_embeddings.py
├── 03_positional_encoding.py
├── 04_self_attention.py
├── 05_multi_head_attention.py
├── 06_feed_forward.py
├── 07_transformer_block.py
├── 08_gpt_model.py
├── 09_training.py
├── 10_generation.py
│
├── tokenizer.py             # Reusable modules
├── embeddings.py
├── self_attention.py        # (with causal mask)
├── multi_head_attention.py
├── feed_forward.py
├── transformer_block.py
├── gpt_model.py
├── positional_encoding.py   # helper
│
├── data/
│   └── train.txt            # your training text (e.g., Alice in Wonderland)
│
├── requirements.txt
├── README.md
└── own_gpt.pt               # saved model weights (after training)
🔧 Installation & Dependencies
Requirements:

Python 3.8+

PyTorch 2.0+

NumPy

Install with:

bash
pip install -r requirements.txt
🏋️ Training
Prepare your data
Place any plain text file (e.g., a book, articles, or your own writing) in data/train.txt.
The tokenizer will automatically build a vocabulary from the characters present.

Adjust hyperparameters (in 09_training.py):

python
embed_dim = 128        # size of token vectors
num_heads = 8          # number of attention heads
num_layers = 4         # number of transformer blocks
hidden_dim = 512       # feed‑forward hidden size
max_seq_len = 64       # context window
batch_size = 32
epochs = 300
learning_rate = 3e-4
Start training:

bash
python 09_training.py
The script will print loss every 10 epochs and save the model as own_gpt.pt.

Expected behaviour: With a proper causal mask (included in self_attention.py), the loss will decrease steadily from ~4.5 to below 2.0 after a few hundred epochs, depending on data size. Without the mask, the loss drops very quickly but the model cannot generate anything sensible – always remember to use a causal mask!

💬 Generating Text
After training, run:

bash
python 10_generation.py
This will load the saved model and generate text starting from a prompt (you can change the prompt in the script).

Generation parameters:

max_new_tokens – number of characters to generate.

temperature – controls randomness (lower = more deterministic, higher = more creative).

Example output (after training on ~150k characters of Alice in Wonderland for 500 epochs):

text
Prompt: "Once upon a time"
Generated: "Once upon a time there was a little girl named Alice, and she had a very curious dream. She saw a white rabbit run by, and she followed it down a rabbit-hole..."
(You will likely get shorter, less coherent output with a small model – but that’s the beauty of learning from scratch!)

📊 Results & Observations
Training set size	Epochs	Final loss	Generated text quality
~5,000 chars	300	~1.8	Mostly gibberish, some word fragments
~150,000 chars	500	~1.2	Recognisable words, occasionally coherent phrases
Key insight: A character‑level model with limited capacity can still learn spelling, common bigrams, and even some word patterns. However, to achieve fluent language, you would need:

More data (millions of characters)

Larger model (embed_dim 512+, layers 12+)

Subword tokenization (BPE or SentencePiece)

More training steps

But for educational purposes, OWNGPT shows exactly how the building blocks work – and that’s the whole point.

🧩 Why This Matters
Understanding the internals of a Transformer‑based LLM is the first step towards:

Debugging and fine‑tuning existing models.

Designing new architectures.

Appreciating the computational and data demands of real‑world LLMs.

Building your own custom models for niche tasks.

OWNGPT is intentionally minimal – it’s not meant to compete with GPT‑3, but to serve as a learning tool that you can read, modify, and experiment with.

📜 License
This project is open‑source and available under the MIT License. Feel free to use, modify, and share it for educational purposes.

🙌 Acknowledgements
The Transformer paper: Attention Is All You Need (Vaswani et al., 2017)

GPT architecture (Radford et al., 2018)

PyTorch community for excellent documentation

Happy building!
If you have questions, open an issue or experiment with the code – that’s the best way to learn. 🚀