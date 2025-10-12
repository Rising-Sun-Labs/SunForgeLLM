# LLM Terminology & Concepts — Comprehensive Glossary

> A practical, high‑coverage glossary of modern Large Language Model (LLM) terms. Grouped by topic, includes acronyms, common synonyms, and quick notes. (You won’t *literally* find **every** term in the universe, but this aims to be exhaustive for day‑to‑day engineering, research papers, and productization.)

---

## 1) Core Concepts

- **AI / ML / DL** — Artificial Intelligence / Machine Learning / Deep Learning.
- **Neural Network (NN)** — Parametric function approximator trained with data.
- **Parameter / Weight** — Trainable scalar in a model; millions to trillions.
- **Embedding** — Dense vector representation of tokens, text, images, etc.
- **Hidden State (Activation)** — Intermediate layer output inside a network.
- **Loss Function** — Objective to minimize during training (e.g., cross‑entropy).
- **Gradient / Backpropagation** — Derivatives of loss w.r.t. parameters; used to update weights.
- **Batch / Minibatch** — Set of samples processed together for efficiency.
- **Epoch / Step** — One pass over data / one optimizer update respectively.
- **Overfitting / Underfitting** — Too specific to training data / not learning enough patterns.
- **Generalization** — Performance on unseen data.
- **Transfer Learning** — Reusing knowledge from one task/domain to another.

---

## 2) Language Modeling Basics

- **Language Model (LM)** — Model that assigns probabilities to sequences of tokens.
- **Causal LM / Autoregressive LM** — Predicts next token given previous tokens.
- **Masked LM** — Predicts masked tokens (e.g., BERT) — *not typically used for generation*.
- **Seq2Seq** — Encoder–decoder model for conditional generation (e.g., translation).
- **Perplexity (PPL)** — Exponential of average negative log‑likelihood; lower is better.
- **Next‑Token Prediction (NTP)** — Standard pretraining objective for causal LMs.
- **Teacher Forcing** — Feeding ground‑truth tokens at training time.
- **Exposure Bias** — Train/test mismatch when model sees its own outputs at inference.
- **Beam Search / Greedy / Sampling** — Decoding strategies for token generation.
- **Top‑k / Top‑p (Nucleus) Sampling** — Truncate distribution to the most likely k tokens / smallest set whose cumulative mass ≥ p.
- **Temperature** — Flattens or sharpens token distribution during sampling.

---

## 3) Tokenization & Text Processing

- **Token** — Smallest processed unit (subword, byte, character, word).
- **Vocabulary (Vocab)** — Set of tokens known to the model.
- **BPE (Byte‑Pair Encoding)** — Subword tokenizer merging frequent pairs.
- **Unigram LM Tokenizer** — Probabilistic subword segmentation approach.
- **WordPiece** — Subword method used in early models like BERT.
- **SentencePiece** — Library for BPE/Unigram tokenization, language‑agnostic.
- **Byte‑Level Tokenization / Byte‑Fallback** — Robust handling of any text/UTF‑8.
- **OOV (Out‑of‑Vocabulary)** — Unknown tokens; typically mitigated by subword/byte tokenizers.
- **Special Tokens** — e.g., `<bos>`, `<eos>`, `<pad>`, `<unk>`, role tokens (`<sys>`, `<usr>`, `<asst>`).

---

## 4) Transformer Architecture

- **Transformer** — Architecture based on attention; replaces recurrence/convolutions.
- **Encoder / Decoder** — Encoder reads full context; decoder generates autoregressively.
- **Self‑Attention** — Computes token‑token interactions within a sequence.
- **Multi‑Head Attention (MHA)** — Multiple attention heads in parallel.
- **Q / K / V (Query/Key/Value)** — Linear projections used to compute attention.
- **Scaled Dot‑Product Attention** — Softmax(QKᵀ/√d) V formulation.
- **Causal Mask** — Prevents access to future tokens (strictly autoregressive).
- **FFN / MLP Block** — Position‑wise feed‑forward subnetwork (often SwiGLU/GeGLU).
- **LayerNorm / RMSNorm** — Normalization layers for stability.
- **Residual Connection (Skip Connect)** — Adds input to output of a block.
- **Position Embedding** — Injects order information; absolute, sinusoidal, learned.
- **RoPE (Rotary Position Embedding)** — Rotation in complex plane to encode position.
- **ALiBi / Relative Position Bias** — Alternative positional schemes (Attention with Linear Biases).
- **SwiGLU / GeGLU** — Gated activation variants improving Transformer FFNs.
- **KV Cache** — Cached Key/Value states for faster autoregressive inference.
- **GQA / MQA** — Grouped/Multiple Query Attention to optimize memory/latency.
- **Windowed / Sliding‑Window Attention** — Attend to a recent window for long contexts.
- **Sparse / Block‑Sparse Attention** — Reduces quadratic attention cost.
- **Mixture‑of‑Experts (MoE)** — Conditional compute; routes tokens to expert MLPs.
- **Speculative Decoding** — Draft model proposes tokens; target model verifies to speed up decoding.
- **Prefix‑LM** — Allows a bidirectional “prefix” and autoregressive tail.

---

## 5) Optimization & Training

- **SGD / Adam / AdamW** — Optimizers; AdamW decouples weight decay.
- **Learning Rate (LR) Schedule** — Warmup, cosine decay, linear decay, etc.
- **Gradient Clipping** — Prevent exploding gradients by bounding norm.
- **Mixed Precision / FP16 / BF16 / FP8** — Reduced precision for speed/memory.
- **Gradient Checkpointing** — Trade compute for memory by recomputing activations.
- **FSDP / ZeRO** — Fully Sharded Data Parallel / ZeRO memory partitioning (DeepSpeed).
- **Data / Tensor / Pipeline Parallelism** — Split training across data, layers, or model dims.
- **DDP** — Distributed Data Parallel (multi‑GPU training).
- **Loss Scaling** — Stabilizes mixed‑precision training (esp. FP16).
- **Weight Decay** — Regularization via L2 on weights (decoupled in AdamW).
- **EMA (Exponential Moving Average)** — Smoothed parameter copy for evals.
- **Curriculum Learning** — Schedule training difficulty/data order.
- **Early Stopping / Checkpointing** — Stop when no improvement; save periodic snapshots.
- **Throughput (tokens/s)** — Effective training speed metric.
- **Budget (FLOPs)** — Total floating‑point operations needed; guides compute planning.

---

## 6) Regularization & Stabilization

- **Dropout / Stochastic Depth** — Random deactivation of neurons/layers.
- **Label Smoothing** — Distributes probability mass to non‑targets slightly.
- **Weight Tying** — Share embedding and output layer weights.
- **Initialization Schemes** — Xavier, Kaiming, µParam, etc.
- **Norm Clipping / Gradient Noise** — Stabilization heuristics.

---

## 7) Data Engineering for LLMs

- **Corpus / Dataset / Mixture** — Combined sources of training text/code/etc.
- **Deduplication (Exact / Near‑Dup)** — Remove duplicates using hashing/SimHash/MinHash.
- **Filtering** — Language ID, NSFW/toxicity, PII removal, license filters (for code).
- **Sharding** — Splitting datasets into uniformly sized binary shards.
- **Packing** — Concatenating examples to fill fixed sequence length (minimize padding).
- **Curation** — Hand‑select or programmatically choose higher‑quality samples.
- **Synthetic Data** — Data generated by models or templating; requires careful validation.
- **PII / Secrets Scrubbing** — Remove emails, API keys, credentials, etc.
- **Copyright & Licensing** — Ensure legal use; track licenses for code corpora.
- **Data Mixing Ratios** — Proportions for web/code/docs, etc.
- **RNG Seeding / Determinism** — Reproducible data orderings.
- **Telemetry‑Driven Mining** — Use product logs (with privacy controls) to guide data collection.

---

## 8) Instruction Tuning & Alignment

- **SFT (Supervised Fine‑Tuning)** — Train on (prompt, response) pairs to shape behavior.
- **RLHF** — Reinforcement Learning from Human Feedback (PPO‑based).
- **DPO** — Direct Preference Optimization (supervised objective on preference pairs).
- **RRHF** — Rank Responses with Human Feedback (alternative to PPO).
- **Constitutional AI** — Train with principles/policies to reduce harmful outputs.
- **System Prompt** — Hidden instructions controlling assistant behavior.
- **Guardrails** — Policies and mechanisms limiting unsafe outputs.
- **Refusal / Deferral** — Model declines unsafe or out‑of‑scope requests.
- **Critic / Self‑Refine** — Model critiques/edits its own drafts.
- **Function Calling / Tool Use** — Structured call-outs to external functions/APIs.
- **Agent / ReAct** — Reason‑and‑Act prompting; interleave thoughts and actions (tools).

---

## 9) Evaluation & Benchmarks

- **Train / Val / Test Split** — Separate sets to avoid leakage.
- **Perplexity (PPL)** — Common intrinsic metric.
- **Exact Match (EM) / F1** — QA metrics for span extraction/answers.
- **BLEU / ROUGE / METEOR / ChrF** — Text generation metrics (translation/summarization).
- **BERTScore / MoverScore** — Semantic similarity metrics using embeddings.
- **Code Benchmarks** — HumanEval, MBPP, CodeXGLUE, MultiPL‑E, SWE‑bench (variants).
- **Reasoning Benchmarks** — GSM8K, MATH, ARC, HellaSwag, BIG‑bench (BBH), DROP.
- **Hallucination Rate** — Frequency of unsupported or fabricated claims.
- **Toxicity / Safety Metrics** — Measures of harmful content propensity.
- **Latency (p50/p95/p99)** — Response time percentiles in serving.
- **Throughput (req/s, tok/s)** — Serving capacity.
- **A/B Testing** — Online evaluation with user traffic.
- **Golden Sets** — Hand‑curated evaluation items for regression testing.
- **Calibration** — Model confidence alignment with accuracy.

---

## 10) Inference & Serving

- **vLLM / TensorRT‑LLM / llama.cpp** — Popular inference engines/backends.
- **Batching / Continuous Batching** — Merge requests for throughput.
- **KV Cache Reuse** — Reuse attention states across tokens/turns.
- **Streaming (SSE/WS)** — Deliver tokens incrementally to clients.
- **Prompt Caching** — Cache embeddings/KV for repeated prompts.
- **Context Length / Max Tokens** — Bound on input+output tokens.
- **Stop Sequences** — Tokens/strings that end generation.
- **Safety Filters / Moderation** — Pre/post‑processing checks in serving layer.
- **Observability** — Logs, metrics, traces; OpenTelemetry integration.
- **Rate Limiting / Quotas** — Protect service from abuse and control cost.
- **Speculative Decoding** — Latency optimization via draft+target models.
- **Distillation** — Train a smaller/faster model to mimic a larger one.

---

## 11) Compression & Deployment Optimizations

- **Quantization** — Reduce precision (e.g., 8‑bit, 4‑bit, FP8). Static/PTQ or QAT.
- **Pruning** — Remove weights/neurons with little impact.
- **KV Cache Quantization** — Quantize attention cache for memory savings.
- **Low‑Rank Adapters (LoRA/QLoRA)** — Parameter‑efficient finetuning.
- **Knowledge Distillation** — Teacher–student training for smaller models.
- **FlashAttention** — Memory‑efficient attention kernel; speeds training/inference.
- **Paged Attention** — Manage KV cache efficiently (e.g., vLLM).
- **Graph Compilers** — TorchInductor, XLA, TensorRT; fuse ops for speed.

---

## 12) Retrieval & RAG (Retrieval‑Augmented Generation)

- **Retriever** — Finds relevant documents/snippets for a query.
- **Dense Retrieval** — Embedding‑based (bi‑encoder) search (FAISS, ScaNN).
- **BM25 / Sparse Retrieval** — Classical lexical matching with term weighting.
- **Hybrid Retrieval** — Combine dense + sparse signals.
- **Reranker / Cross‑Encoder** — Re‑scores candidates with a deeper model.
- **Context Packing / Chunking** — Split docs into windows; include top‑k in prompt.
- **Citations / Attributions** — Link outputs to sources for grounding.
- **Index** — Data structure storing embeddings (IVF, HNSW, PQ).
- **Freshness / Recency** — Time‑aware retrieval (news, changing docs).
- **Query Expansion** — Reformulate queries to improve recall.
- **Docstore / Vector DB** — Storage layer for documents + embeddings.
- **Hallucination Mitigation** — Use retrieved evidence; answer “I don’t know” when needed.

---

## 13) Multimodality

- **Vision‑Language Model (VLM)** — Jointly processes images and text.
- **ViT (Vision Transformer)** — Transformer for images (patch embeddings).
- **CLIP / SigLIP** — Contrastive vision–text pretraining (aligns modalities).
- **Q‑Former / Perceiver Resampler** — Modules to condense visual tokens.
- **Projector** — Maps vision/audio features into LM hidden space as “visual tokens.”
- **ASR (Automatic Speech Recognition)** — Transcribe audio to text (e.g., encoder–decoder).
- **TTS (Text‑to‑Speech)** — Synthesize voice from text.
- **Multimodal Instruction Tuning** — Supervision with image+text (and audio) prompts.
- **OCR / Doc‑VLM** — Recognize text/structure in documents/PDFs.
- **Temporal Models** — Video understanding; temporal attention.

---

## 14) Prompting Patterns & UX

- **Zero‑Shot / Few‑Shot** — No examples vs. a few in‑prompt examples.
- **Chain‑of‑Thought (CoT)** — Encourage intermediate reasoning steps.
- **Self‑Consistency** — Sample multiple CoT paths and majority‑vote.
- **ReAct** — Interleave reasoning with tool actions.
- **Scratchpad** — Temporary token budget for intermediate computation.
- **Instruction Hierarchy** — System > Developer > User messages (by precedence).
- **Prompt Templates** — Reusable structured prompts for tasks.
- **Stop Words / Banned Tokens** — Prevent certain strings (e.g., secrets).

---

## 15) Safety, Security & Policy

- **Red‑Teaming** — Actively probing for failures (jailbreaks, bias, toxicity).
- **Jailbreak** — Prompt that bypasses guardrails.
- **Prompt Injection** — Malicious instructions inside inputs/RAG docs.
- **Data Exfiltration** — Leaking secrets/PII via model outputs.
- **PII (Personally Identifiable Information)** — Sensitive user data.
- **Content Filters** — Classifiers/rules to block categories (e.g., hate, self‑harm).
- **Watermarking / Provenance** — Identify generated content / trace outputs.
- **Model Stealing / Extraction** — Adversary replicates a model via queries.
- **Adversarial Examples** — Inputs crafted to cause mistakes.
- **Safety Spec / Policy** — Written rules the model/system should follow.
- **Usage Governance** — RBAC, quotas, audit logs, consent tracking.
- **Secure Tooling** — Sandboxes (seccomp, gVisor, Firecracker) for tool execution.

---

## 16) Long‑Context & Memory

- **Context Length** — Maximum tokens the model can condition on.
- **RoPE Scaling (NTK/YaRN)** — Techniques to extend usable context.
- **Long‑Range Attention** — Sparse/sliding/windowed attention.
- **Memory Replay / Summarization** — Compress old context to fit budgets.
- **Retrieval Memory** — External store for past interactions.
- **Continuation Stability** — Model quality doesn’t degrade as context grows.

---

## 17) Distributed Systems & Infra

- **Cluster / Node / GPU / TPU** — Hardware resources for training/inference.
- **Scheduler / Orchestrator** — Slurm, Kubernetes, Ray.
- **Checkpoint / Shard** — Saved weights split across devices/files.
- **Fault Tolerance / Elasticity** — Resume training after failures; scale workers.
- **Throughput vs. Latency** — Bulk speed vs. single‑request speed trade‑off.
- **Autoscaling** — Scale replicas based on load.
- **Cost per 1k Tokens** — Serving economics; budget controls.
- **AIOps / MLOps** — Operational discipline for model lifecycle.

---

## 18) Code Intelligence (LLM‑for‑Code)

- **AST / CFG** — Abstract Syntax Tree / Control Flow Graph.
- **Static Analysis / Linting** — Analyze code without running it.
- **Dynamic Analysis** — Execute tests/instrument to observe behavior.
- **Unit / Integration / E2E Tests** — Hierarchy of software tests.
- **Diff / Patch / PR / MR** — Changesets and review terminology.
- **Refactoring** — Behavior‑preserving code restructuring.
- **Type Inference** — Deduce types (for typed languages).
- **Symbolic Execution** — Explore paths with symbolic inputs.
- **Fuzzing** — Randomized input generation to find bugs.
- **License Compliance** — Respect code licenses in suggestions.
- **Hallucinated APIs** — Nonexistent functions/classes suggested by model.

---

## 19) Math & Notation You’ll See

- **Softmax** — `softmax(zᵢ)=exp(zᵢ)/Σⱼexp(zⱼ)`.
- **Cross‑Entropy** — `−Σ p(x) log q(x)`; for next‑token prediction.
- **Perplexity** — `exp(cross‑entropy)`.
- **Dot Product / Cosine Sim** — Similarity in embedding space.
- **Layer Shapes** — `[Batch, Time, Hidden]`, `[B,T,C]`.
- **Big‑O** — Complexity notation (e.g., attention is O(T²) in sequence length).

---

## 20) Paper & Ecosystem Terms

- **GPT / LLaMA / Mistral / Qwen / PaLM / T5 / BERT** — Well‑known model families (architectural inspirations).
- **FlashAttention / xFormers** — Libraries for fast attention kernels.
- **DeepSpeed / Megatron‑LM** — Training frameworks for large models.
- **vLLM / FasterTransformer / TensorRT‑LLM** — High‑performance inference stacks.
- **FAISS / ScaNN / Milvus** — Vector search/index libraries.
- **OpenTelemetry / Prometheus / Grafana** — Observability stack.
- **Hugging Face Datasets / Tokenizers** — Popular data tooling (even if you roll your own).

---

## 21) Productization & UX

- **SLA / SLO** — Service‑level agreement/objectives (uptime, latency).
- **Cold Start / Warm Pool** — Instance spin‑up vs. pre‑initialized workers.
- **A/B / Bandits** — Online experimentation frameworks.
- **Prompt Library** — Catalog of prompts/templates for teams.
- **Playground** — UI for testing prompts/models.
- **Redlines** — Non‑negotiable safety or compliance constraints.

---

## 22) Common Acronyms (Quick Table)

| Acronym | Expansion | Context |
|---|---|---|
| LM | Language Model | General |
| LLM | Large Language Model | General |
| VLM | Vision‑Language Model | Multimodal |
| RAG | Retrieval‑Augmented Generation | Retrieval |
| SFT | Supervised Fine‑Tuning | Alignment |
| RLHF | Reinforcement Learning from Human Feedback | Alignment |
| DPO | Direct Preference Optimization | Alignment |
| PPO | Proximal Policy Optimization | RL |
| MoE | Mixture of Experts | Architecture |
| MHA | Multi‑Head Attention | Transformer |
| GQA/MQA | Grouped/Multiple Query Attention | Attention scaling |
| RoPE | Rotary Position Embedding | Positional encoding |
| ALiBi | Attention with Linear Biases | Positional bias |
| FP8/BF16/FP16 | Numeric Precisions | Training/Inference |
| FSDP | Fully Sharded Data Parallel | Training |
| ZeRO | Zero Redundancy Optimizer | Memory scaling |
| DDP | Distributed Data Parallel | Training |
| EMA | Exponential Moving Average | Stabilization |
| PPL | Perplexity | Metric |
| EM/F1 | Exact Match / F1 | Eval |
| HNSW/IVF/PQ | Index structures | Vectors |
| KV Cache | Key‑Value Cache | Inference |

---

### Final Notes
- Terminology evolves. New kernels (e.g., FlashAttention‑3), new routing (MoE variants), and fresh alignment methods appear frequently. The categories above give you 95% coverage for building, evaluating, and shipping LLM systems.
