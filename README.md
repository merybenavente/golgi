# Golgi

**The Active Memory Manager for LLM Agents.**

Engineering the illusion of memory requires active management, not just passive storage.

Golgi is a proposal for a new memory management architecture, born from a deep analysis of the current landscape of adaptive memory research (including works like A-Mem, Mem0, and Zep). While most current systems focus on how to store data and how to retrieve it, Golgi focuses on what should be stored, and how to manage it.

In cells, the Golgi apparatus acts as a sorting station, processing and transforming molecules to prepare them for transport out of the cell or storage within the cell's own membranes. Similarly, our Golgi serves as the "Memory Manager" layer, packaging, sorting, and modifying information before storing it in long-term memory.

## Manifesto

I analyzed the current SotA in LLM memory systems. Most operate on a "Storage First" paradigm. They prioritize getting text into the database quickly—usually by chunking it and embedding it—and rely entirely on vector similarity at query time to find relevance.

This approach creates a retrieval bottleneck. To identify a complex pattern or insight, the system is forced to scan and compare large volumes of unstructured chunks at query time. [todo: some of them have some of these mechanism] There is no pre-computation of relationships, no hierarchy, and no concept of "forgetting" irrelevant details.

## The Shift

| Feature | Current SOTA (Storage First) | Golgi (Relationship First) |
|---------|------------------------------|----------------------------|
| **Philosophy** | Dump now, will find later | Sort now, retrieve smarter |
| **Ingestion** | Passive Chunking | Active Atomization & Linking |
| **Compute** | Read-Time (Heavy Retrieval) | Write-Time (Smart Storage) |
| **Maintenance** | None (Static) | Dynamic Decay & Consolidation |

## Core Principles

### 1. Structure over Similarity

While vector databases are excellent for finding semantic similarity (things that sound alike), they struggle with structural context (things that relate logically). Golgi proposes managing "Atomic Notes" stored in a highly browseable graph, where vectors are used for indexing, but primary navigation happens through explicit edges (relationships). This allows the manager to traverse from high-level summaries down to specific details.

### 2. Write-Time Compute

Ingestion in Golgi is an active process. When the system receives new information, it doesn't just save it. The Manager parses the input to:
- **Extract**: select only the information worth sharing.
- **Atomize**: Break the stream of text into distinct, self-contained concepts.
- **Link**: [todo]
- **Store in its place**: [todo]

### 3. Memory Consolidation

Information shouldn't live forever with equal weight. Golgi implements a background maintenance process—similar to biological memory consolidation, one that actively decays the relevance of nodes over time. Low-relevance nodes are eventually pruned or summarized into higher-level abstractions, keeping the retrieval space clean.

## Roadmap (in case i end up building in public)

### Stage 0: Analysis & Vision (Completed)

I have completed a review of the current adaptability research and SOTA systems. This phase identified the critical gap between static retrieval systems and the need for dynamic, evolving memory. The vision for Golgi is set: a Memory Manager that self-organizes and prioritizes "active" knowledge.

### Stage 1: Memory extraction

What's the definition of "relevant" on this context?

### Stage 2: Ontology & Schema Definition

Define the data contract. What's an "atomic memory", and how do they define "relationships" between them. This schema is critical because it stablished the foundation of the system.

### Stage 4 and beyond: tbd
Once the evaluation over the static graph memroy seams in place from a human POV, I would like to test how a layer of vector retrieval can navigate it.
From there, I should also explore the decay architectures I have in mind.

## Contributing

This is currently a solo project where I am building in public.
I am documenting the process of moving from research synthesis to working code.