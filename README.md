
Vision Transformer (ViT) - A Detailed Explanation

The Vision Transformer (ViT) is a deep learning architecture that applies the Transformer model (originally developed for Natural Language Processing) to computer vision tasks. Unlike traditional Convolutional Neural Networks (CNNs) that rely on convolutional filters, ViT processes images as sequences of patches and applies self-attention to capture relationships between these patches.
1. How ViT Works Internally?

Step 1: Splitting an Image into Patches

Unlike CNNs that process entire images using convolutional kernels, ViT splits an image into non-overlapping patches.

Example:

Suppose we have a 224×224 RGB image. If we use 16×16 patches, we will have:

(224/16)² = 14 × 14 = 196 patches

Each patch has 16×16×3 (RGB) pixels and is flattened into a vector of size 768 (for ViT-Base).

Step 2: Patch Embeddings

Each flattened patch is projected into a D-dimensional vector (e.g., 768 for ViT-Base) using a learnable linear transformation (fully connected layer).



E = Wp × Pi



where:

*   `Wp` is a learnable weight matrix,
*   `Pi` is the i-th flattened patch,
*   `E` is the patch embedding.

Step 3: Adding Positional Embeddings

Unlike CNNs, Transformers do not inherently capture spatial information. To retain the positional structure of the patches, we add learnable positional embeddings to the patch embeddings.



Z = E + PE



where:

*   `Z` is the final input embedding to the Transformer,
*   `PE` is the positional encoding.

Why do we need Positional Embeddings?

Transformers treat inputs as sequences, but an image has a fixed spatial structure. Adding positional information helps the model understand how patches relate to each other.

Step 4: Transformer Encoder

The Transformer encoder consists of L layers (e.g., 12 layers in ViT-Base). Each layer has:

*   Multi-Head Self-Attention (MHSA)
*   Feed-Forward Network (FFN)
*   Layer Normalization
*   Skip (Residual) Connections

4.1. Multi-Head Self-Attention (MHSA)

The self-attention mechanism allows each patch to interact with every other patch, capturing global dependencies.

Self-Attention Computation:**

1.  Compute Query (Q), Key (K), and Value (V) Matrices:
    ```
    Q = XWq, K = XWk, V = XWv
    ```
2.  Compute Attention Scores:
    ```
    A = softmax(QKᵀ / √dk)
    ```
3.  Compute Weighted Sum of Values:
    ```
    Z = A × V
    ```

Key Advantages of Self-Attention:**

*   Captures long-range dependencies
*   Learns contextual relationships across the image

4.2. Feed-Forward Network (FFN)

Each attention output is passed through a feed-forward network:



FFN(X) = max(0, XW₁ + b₁)W₂ + b₂



This helps transform feature representations non-linearly.4.3. Layer Normalization & Skip Connections

Layer Normalization stabilizes training. Skip (Residual) Connections help prevent vanishing gradients.

Transformer Block (Repeated L Times):**

Self-Attention → Add & Norm → Feed-Forward → Add & Norm

Step 5: Class Token for Classification

ViT introduces a special `[CLS]` token (similar to BERT) that learns a global representation of the image. The final `[CLS]` token embedding is passed to a fully connected layer (MLP Head) for classification.

2. ViT Model Variants

There are multiple ViT architectures, varying in size:

| Model        | Layers (L) | Hidden Size (D) | Heads (H) | Params |
|--------------|------------|-----------------|-----------|--------|
| ViT-B (Base) | 12         | 768             | 12        | 86M    |
| ViT-L (Large)| 24         | 1024            | 16        | 307M   |
| ViT-H (Huge) | 32         | 1280            | 16        | 632M   |

3. ViT vs. CNNs: Key Differences

| Feature            | ViT                               | CNN                                  |
|--------------------|------------------------------------|--------------------------------------|
| Feature Extraction | Uses Self-Attention                | Uses Convolutions                     |
| Spatial Information| Requires Positional Encoding         | Captures spatial hierarchy naturally    |
| Computational Cost | Higher (quadratic complexity)       | Lower (optimized convolutions)        |
| Performance on Small Datasets | Requires pretraining                | Works well with limited data           |
| Global Dependencies| Captures global relationships        | Focuses on local features             |
| Data Efficiency    | Needs large datasets                 | Works better with fewer samples        |

## 4. Advantages of Vision Transformers

*   **Better for Large Datasets:** Outperforms CNNs when trained on datasets like ImageNet-21k.
*   **Captures Long-Range Dependencies:** Self-attention connects distant patches, unlike CNNs.
*   **Scalability:** Works well with transfer learning.
*   **Interpretable:** Attention maps show which parts of an image the model focuses on.

## 5. Challenges of ViT

*   **High Computational Cost:** Needs quadratic attention computations for large images.
*   **Requires Large Datasets:** Performs poorly on small datasets unless pretrained.
*   **Less Robust to Small Data Shifts:** CNNs generalize better when dataset size is limited.


SOTA (State-of-the-Art) Methods:
State-of-the-Art (SOTA) methods refer to the most advanced, high-performing techniques available in a particular field. These methods represent the best-known solutions based on current research, innovations, and real-world applications.

SOTA techniques are constantly evolving as new discoveries and optimizations emerge, improving accuracy, efficiency, and overall performance across different domains like machine learning, deep learning, computer vision, natural language processing (NLP), reinforcement learning, and more.

1. SOTA in Machine Learning & Deep Learning
Machine learning (ML) and deep learning (DL) have rapidly advanced, with SOTA models achieving superior performance in complex tasks.

🔹 Key SOTA Models in Deep Learning
Model	Use Case	Description
GPT-4, Llama 3, Claude	NLP (Text Generation)	These transformer-based models generate human-like text, answer questions, and assist with coding.
Stable Diffusion, Midjourney, DALL·E 3	Image Generation	AI models that generate realistic images from text prompts using diffusion models.
Vision Transformers (ViT)	Image Classification	A deep learning model that outperforms CNNs by applying transformer architectures to images.
YOLOv8, DETR	Object Detection	Advanced models for real-time object detection in images and videos.
Whisper (by OpenAI)	Speech Recognition	A highly accurate speech-to-text model trained on diverse multilingual data.
2. SOTA in Natural Language Processing (NLP)
NLP has seen remarkable advancements with transformer models and large language models (LLMs).

🔹 Key SOTA NLP Methods
Method	Use Case	Description
BERT, RoBERTa, T5	Text Classification, Sentiment Analysis	Pretrained models used for understanding text, classifying sentiments, and answering questions.
ChatGPT, Claude, Gemini	Conversational AI	Advanced chatbots that can generate human-like responses in various contexts.
mT5, BLOOM, XLM-R	Multilingual NLP	Language models that support multiple languages for translation, summarization, and text understanding.
PaLM 2, GPT-4 Turbo	Code Generation	AI models capable of writing, debugging, and explaining code.
3. SOTA in Computer Vision (CV)
Computer vision has seen breakthroughs with transformers, CNNs, and GANs.

🔹 Key SOTA Methods in CV
Method	Use Case	Description
ResNet, EfficientNet	Image Classification	Deep learning architectures for recognizing objects in images.
StyleGAN, Stable Diffusion	AI Art Generation	Models that create realistic or artistic images from noise and text prompts.
Segment Anything Model (SAM)	Image Segmentation	A powerful AI model for detecting and segmenting objects in images.
DINOv2	Self-Supervised Learning	Used for feature extraction and object recognition without labeled data.
4. SOTA in Reinforcement Learning (RL)
Reinforcement Learning (RL) has led to superhuman AI systems capable of playing games, controlling robots, and optimizing real-world processes.

🔹 Key SOTA RL Models
Model	Use Case	Description
AlphaGo, AlphaZero	Board Games (Chess, Go)	AI models that learned to play and defeat world champions in strategy games.
MuZero	Generalized RL	A self-learning model that can play multiple games without prior rules.
Deep Q-Networks (DQN), PPO	Robotics & Automation	Used in robotics, autonomous driving, and decision-making tasks.
5. SOTA in Speech and Audio Processing
Speech recognition and synthesis have been transformed by deep learning.

🔹 Key SOTA Methods in Speech & Audio
Method	Use Case	Description
Whisper (OpenAI)	Speech-to-Text	A highly accurate model for transcribing spoken words into text.
WaveNet, VITS	Text-to-Speech (TTS)	AI models that generate natural-sounding speech from text.
DeepSpeech (Mozilla)	Speech Recognition	A deep learning model used for automatic speech transcription.
6. How Are SOTA Methods Identified?
To determine whether a method is state-of-the-art, researchers use benchmarks and leaderboards:

🔹 Key Benchmarks for SOTA Models
Benchmark	Field	Description
GLUE, SuperGLUE	NLP	Tests text understanding and classification models.
COCO, ImageNet	Computer Vision	Evaluates object detection and classification accuracy.
Librispeech	Speech Recognition	Measures the performance of speech-to-text models.
MS MARCO, SQuAD	Question Answering	Assesses how well models answer factual questions.
