# **AI Scientist Challenge – Evaluator Guide (Version 2025.11.12)**

## **1. Overview**

The **AI Scientist Challenge** is divided into four tracks:

1. **Literature Review** — automated academic survey and synthesis
2. **Paper QA** — scientific question answering grounded in a research paper
3. **Ideation** — generation of novel and feasible research ideas
4. **Paper Review** — automated critical review and scoring of papers

Unlike traditional benchmarks that rely on static test datasets, this competition is evaluated dynamically through the **AI Scientist Challenge Arena**, an interactive platform designed to assess model performance under realistic conditions. The **Arena platform** features:

- **Pairwise blind evaluation** — Each evaluator first **inputs a custom query** through the Arena interface (for example, a literature review topic or a request for idea generation). The Arena then automatically sends this query to **two anonymous models** (Model A and Model B) and returns their respective outputs side by side.
   Evaluators are asked to **cast a vote** indicating which response they prefer — **A**, **B**, **Tie** or **Both are bad** — based on the answer quality.
- **Elo-based ranking mechanism** — System scores are updated dynamically based on pairwise outcomes, producing an evolving real-time leaderboard.

Evaluators can access the Arena through the following link:
 👉 **http://39.97.229.86/leaderboard**

All evaluators have been issued unique access tokens for secure login and tracking.

## **2. Evaluation Guidelines**

### 2.0 **General instructions** applicable to all tracks:

1. **Time Limits and Output Truncation**
    Each track has its own time limit:

   - Literature Review – **15 minutes**
   - Paper QA – **15 minutes**
   - Ideation – **10 minutes**
   - Paper Review – **20 minutes**

   When the time limit is reached, the Arena front-end will **automatically stop displaying new output**, which may result in truncated responses.

2. **Persistent Chat Windows and Parallel Evaluation**
    Each Arena chat window is **persistent**, meaning that once a query is submitted, the response will continue generating in the background even if you leave the page.
    Evaluators **do not need to stay on the same chat tab** during generation — you may immediately open a **new chat** and submit the next query.
    This allows multiple queries to run in parallel, saving time. After all responses have finished, evaluators can return to each tab to review and score them.

### **2.1 Track A – Literature Review**

#### **Goal**

Evaluate the model’s ability to perform comprehensive and accurate literature review on a specified research topic.

#### **Example Queries**

- Please help me comprehensively review recent progress in multimodal large language models.
- Summarize the key methods and challenges in AI Scientists.
- Provide an overview of recent trends in world models.
- Please summarize recent works combining reinforcement learning with LLM reasoning agents.
- Review how AI techniques are being used in climate modeling and sustainability research.
- Summarize the development and evolution of RL algorithms for LLMs.
- Review the latest research in the field of symbolic regression since 2024.
- Compile all works published in Nature, Science, and their sub-journals over the past five years that focus on the intersection of AI and chemistry.
- Trace the research progress on test-time scaling presented at the top three machine learning conferences over the past two years.
- Summarize the state-of-the-art in AI-assisted drug discovery and molecular design.

#### **Evaluation Criteria**

| Dimension          | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| **Coverage**       | Does the response capture major works and directions in the field? |
| **Accuracy**       | Are cited works and summaries factually correct?             |
| **Organization**   | Is the review logically structured (e.g., by theme or timeline)? |
| **Insightfulness** | Does it synthesize key insights or trends beyond surface summarization? |
| **Clarity**        | Is the text readable, coherent, and well-formatted?          |

------

### **2.2 Track B – Paper QA**

#### **Goal**

Assess the model’s understanding of a scientific paper and its ability to answer a technical question grounded in the provided PDF.

#### **Example Queries**

- Please carefully analyze and explain the reinforcement learning training methods used in this article.
- What are the main contributions and limitations of this work?
- Explain how this paper’s experiment design supports its conclusions.
- Please summarize the theoretical framework and key assumptions underlying this paper.
- Analyze the data collection and preprocessing methods used and discuss their potential impact on the results.
- Explain how the proposed algorithm or model architecture improves upon prior approaches.
- Identify and evaluate the main sources of uncertainty or potential bias in the experiments.
- Describe how the authors validated their hypotheses and whether the evidence is sufficient.
- Summarize the quantitative results and metrics used, and discuss whether they support the conclusions.
- Analyze the core difference of the proposed method compared to related works.
- Explain any novel evaluation protocols introduced in the paper and their significance.
- Evaluate whether the figures, tables effectively communicate the results.

#### **Evaluation Criteria**

| Dimension         | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| **Understanding** | Does the model correctly interpret the paper’s content?      |
| **Depth**         | Does it go beyond surface-level restatement to provide reasoning or critique? |
| **Relevance**     | Is the answer directly addressing the given question?        |
| **Faithfulness**  | Are claims grounded in the actual text (not hallucinated)?   |
| **Clarity**       | Is the response well-written and logically structured?       |

------

### **2.3 Track C – Ideation**

#### **Goal**

Evaluate the system’s creativity and feasibility in proposing **novel scientific ideas** related to a given topic.

#### **Example Queries**

- Propose a novel research idea that integrates graph neural networks with protein folding simulations.
- Suggest a new benchmark framework for evaluating creativity in LLM-generated scientific hypotheses.
- Propose an approach to enhance data efficiency in molecular property prediction using multimodal reasoning.
- Imagine a future AI scientist capable of performing both experimental control and theoretical inference — outline its core architecture.
- How can physical world constraints be better incorporated into world models for robotic control and simulation?
- Propose a method to improve temporal consistency in multimodal LLM reasoning when processing video and text streams simultaneously.
- How might symbolic reasoning be integrated with large language models to enhance mathematical interpretability in scientific predictions?
- Suggest a technique for real-time adaptive speech synthesis that leverages reinforcement learning with world model predictions.
- Propose a framework to combine proprioceptive feedback and visual input in robot navigation using multimodal world models.
- How can causal structure learning be incorporated into world models to improve long-term prediction accuracy in dynamic physical environments?
- Suggest an approach for cross-modal alignment in LLMs that allows joint reasoning over text, audio, and video signals for scientific hypothesis generation.
- How might energy-efficient constraints be encoded in reinforcement learning agents for real-world robotics experiments?
- Propose a method to enhance explainability of learned physics models in multimodal AI systems using symbolic or rule-based representations.
- How can hierarchical world models be designed to enable reasoning over both micro-scale simulations (e.g., molecules) and macro-scale phenomena (e.g., materials or ecosystems)?

#### **Evaluation Criteria**

| Dimension               | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| **Originality**         | Is the idea non-trivial and creative?                        |
| **Feasibility**         | Could the idea realistically be implemented or tested?       |
| **Scientific Value**    | Does it advance understanding or methodology?                |
| **Clarity & Structure** | Is the proposal coherent and well-motivated?                 |
| **Integration**         | Does it effectively combine reasoning and relevant prior knowledge? |

------

### **2.4 Track D – Paper Review**

#### **Goal**

Evaluate the model’s ability to act as a **reviewer**, providing both structured critique and quantitative scores.

#### **Note:**

**For the Paper Review track, evaluators only need to upload the PDF; no user query input is required.** The review output format is strictly defined as follows:

- Summary
- Strengths
- Weaknesses / Concerns
- Questions for Authors
- Scores:
  - Overall (10)
  - Novelty (10)
  - Technical Quality (10)
  - Clarity (10)
  - Confidence (5)

#### **Evaluation Criteria**

| Dimension           | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| **Expertise**       | Does the response demonstrate a solid understanding of the relevant research area and provide professional feedback? |
| **Score Coherence** | Are numeric scores consistent with textual reasoning?        |
| **Balance**         | Are both strengths and weaknesses addressed fairly?          |
| **Clarity**         | Are the review comments clear, well-structured, and easy to read and understand? |

