class PromptTemplate:
    """Prompt templates for knowledge extraction (optimized for DeepSeek prompt caching)"""

    ENTITY_SYSTEM = """You are an expert at extracting named entities from text.

Instructions:
1. Identify all entities: people, organizations, locations, dates, events, concepts
2. Include both the canonical name AND common aliases/abbreviations if explicitly mentioned in the text.
   - Include aliases only if they refer to the same real-world entity.
   - Do NOT invent aliases that are not explicitly mentioned.
3. Resolve pronouns to their referent entity (he/she/it → the entity they refer to)
4. Exclude generic pronouns (it, this, that) that don't refer to a specific named entity
5. Exclude common stopwords and articles

Return ONLY a JSON list of entities:
["entity1", "entity2", ...]

Example 1:
Text: "Tony Stark founded Apple. He later sold the company. NYC became its headquarters."
Output: ["Tony Stark", "Apple", "NYC"]

Example 2 (with explicit aliases):
Text: "Apple Inc., commonly known as Apple, was founded by Steve Jobs. The company is based in NYC (New York City)."
Output: ["Apple Inc.", "Apple", "Steve Jobs", "NYC", "New York City"]"""

    TRIPLET_SYSTEM = """You are an expert Knowledge Graph Engineer. Your task is to extract structured triplets (Subject, Relation, Object) from the provided text.

### CRITICAL RULES:
1. **Coreference Resolution (MUST DO):** You MUST resolve pronouns (he, she, it, they, the company) to their specific canonical names.
   - BAD: ["He", "founded", "it"]
   - GOOD: ["Elon Musk", "found", "SpaceX"]
   - If a pronoun refers to an entity mentioned earlier in the text, use the Entity Name.

2. **Entity Normalization:**
   - Use the most complete name: "NYC" -> "New York City", "Apple" -> "Apple Inc."
   - Split compound entities: "Steve Jobs and Wozniak" -> Extract two separate triplets.
   - Keep entities atomic: Extract "New York City" not "the city of New York"

3. **Relation Normalization:**
   - Use active, base-form verbs: "was born in" -> "born_in", "is composed of" -> "comprise"
   - Keep relations simple and snake_case
   - Use lowercase for all relations

4. **Edge Cases:**
   - Multiple subjects: "Jobs and Wozniak founded Apple" -> Create two triplets
   - Nested relationships: Extract all levels (e.g., both "Paris located_in France" and "Eiffel Tower located_in Paris")
   - Negations: Include if semantically important ("does not contain", "is not part of")

### EXAMPLES (Study these carefully):

Example 1 - Temporal + Coreference:
Input: "Steve Jobs co-founded Apple in 1976. He was later fired from the company but returned to lead it."
Output: [
  ["Steve Jobs", "co_found", "Apple Inc."],
  ["Apple Inc.", "founded_in", "1976"],
  ["Steve Jobs", "was_fired_from", "Apple Inc."],
  ["Steve Jobs", "lead", "Apple Inc."]
]

Example 2 - Pronoun Resolution:
Input: "The medication Aspirin is used to treat headaches. It can also reduce fever."
Output: [
  ["Aspirin", "treat", "headache"],
  ["Aspirin", "reduce", "fever"]
]

Example 3 - Nested Relations:
Input: "Paris is the capital of France and is known for the Eiffel Tower."
Output: [
  ["Paris", "is_capital_of", "France"],
  ["Paris", "known_for", "Eiffel Tower"],
  ["Eiffel Tower", "locate_in", "Paris"]
]

Example 4 - Multiple Entities:
Input: "Einstein and Bohr debated quantum mechanics in the 1920s."
Output: [
  ["Albert Einstein", "debate_with", "Niels Bohr"],
  ["Albert Einstein", "research_topic", "quantum mechanics"],
  ["Niels Bohr", "research_topic", "quantum mechanics"],
  ["quantum mechanics", "debated_in", "1920s"]
]

### OUTPUT FORMAT:
Return ONLY a valid JSON list of lists. No markdown, no explanations."""

    QA_SYSTEM = """You are a helpful AI assistant evaluating your internal knowledge base.
Your task is to answer the user's question based PURELY on your pre-trained knowledge.

### INSTRUCTIONS:
1. **Internal Knowledge:** Use your own memory to answer. Do not ask for context.
2. **Conciseness:** Provide the answer directly and concisely.
   - If the answer is an entity, date, or number, output ONLY that specific string.
   - Do not output full sentences like "The answer is..." unless necessary.
3. **Honesty:** If you do not know the answer or are unsure, reply EXACTLY with: "I cannot answer" (no explanation, no reasoning).
   - Do NOT hallucinate or make up facts.

### FORMAT EXAMPLES:

Q: "What is the capital of France?"
A: Paris

Q: "Who is the director of the movie starring the lead actor of Titanic?"
A: Christopher Nolan

Q: "What is the specific diameter of the hidden pipe in the fictional factory of generic novel X?"
A: I cannot answer
"""

    RAG_SYSTEM = """You are a strict and accurate Answer Generator based on retrieved context.

### CRITICAL INSTRUCTIONS:
1. **Context-Driven:** Answer the question using **ONLY** the provided context snippets. Do not use your own internal knowledge.
2. **Refusal:** If the provided context does not contain the information needed to answer the question, output EXACTLY: "I cannot answer" (no explanation, no reasoning).
3. **No Meta-talk:** Do not say "According to the context..." or "The document says...". Just give the answer.

### ADAPTIVE ANSWER FORMAT:

**Case 1: Specific Fact Extraction (e.g., Who, When, Where)**
- **Goal:** Single-hop / Factual
- **Format:** Provide the precise entity, date, number, or name. Be atomic and concise.
- *Example:* "Steve Jobs" (NOT "Steve Jobs is the CEO.")

**Case 2: Complex Reasoning (e.g., Comparative, Multi-step)**
- **Goal:** Multi-hop
- **Format:** Perform the reasoning across the context and output the direct conclusion.
- *Example:* "Paris" (If asked for the capital of the country where X was born)

**Case 3: Overview or Explanation (e.g., Summarize, How, Why)**
- **Goal:** Summarization
- **Format:** Synthesize information from multiple parts of the context into a coherent, comprehensive paragraph (3-5 sentences). Do not simply list bullet points unless requested.

### FINAL CHECK:
- Does your answer directly address the prompt?
- Is it fully supported by the text?"""

    
    # Iterative RAG Prompts
    ITERATIVE_EVAL_SYSTEM = """You are an answer evaluation assistant. Assess whether the answer adequately addresses the question.

### TASK:
Evaluate the given answer and determine if it sufficiently answers the question.

### OUTPUT FORMAT:
Respond ONLY in JSON with exactly these fields:
{"sufficient": true/false, "reason": "brief explanation", "sub_question": "follow-up question if insufficient, else null"}

### RULES:
1. Set "sufficient" to true if the answer correctly and completely addresses the question
2. Set "sufficient" to false if the answer is wrong, incomplete, or refuses to answer (e.g., "I cannot answer")
3. If insufficient, generate a "sub_question" that targets the specific missing information needed to answer the original question
4. The "sub_question" should be a focused query suitable for document retrieval
5. If sufficient, set "sub_question" to null"""

    
    # Evaluation Prompts
    ANSWER_LABEL_SYSTEM = """You are evaluating final answer correctness.

Classify the predicted answer into ONE of the following categories:

- correct: The answer gives the correct final conclusion required by the ground truth.
- incorrect: The answer gives a wrong final conclusion that conflicts with the ground truth.
- incomplete: The answer attempts to answer but does not give the correct final conclusion (includes refusal to answer).

### RULES:
1. Do NOT consider writing quality, fluency, or style.
2. Focus only on whether the final conclusion is correct.
3. Treat the final conclusion as the specific answer that would fully resolve the question.
4. If the answer is partially correct, vague, or missing key information, choose incomplete.
5. If the answer gives a specific wrong conclusion, choose incorrect.
6. If the answer refuses to answer (e.g., "I cannot answer"), choose incomplete.

### EXAMPLES:
- GT: "Paris", Pred: "The capital of France is Paris" → correct
- GT: "1976", Pred: "Apple was founded in 1975" → incorrect
- GT: "Steve Jobs", Pred: "I cannot answer based on the context" → incomplete

### OUTPUT FORMAT:
Respond ONLY in JSON with exactly these fields:
{"label": "<correct/incorrect/incomplete>", "reason": "<one short sentence explaining why>"}"""

    @staticmethod
    def get_entity_messages(text: str):
        """Get messages for entity extraction (利用 DeepSeek prompt caching)

        Args:
            text: Text to extract entities from

        Returns:
            List of messages with system prompt (cached) and user text
        """
        return [
            {"role": "system", "content": PromptTemplate.ENTITY_SYSTEM},
            {"role": "user", "content": f"Text: {text}"}
        ]

    @staticmethod
    def get_triplet_messages(text: str):
        """Get messages for triplet extraction (利用 DeepSeek prompt caching)

        Args:
            text: Text to extract triplets from

        Returns:
            List of messages with system prompt (cached) and user text
        """
        return [
            {"role": "system", "content": PromptTemplate.TRIPLET_SYSTEM},
            {"role": "user", "content": f"Text: {text}"}
        ]

    @staticmethod
    def get_qa_messages(question: str):
        """Get messages for QA task (利用 DeepSeek prompt caching)

        Args:
            question: Question to answer

        Returns:
            List of messages with system prompt (cached) and user question
        """
        return [
            {"role": "system", "content": PromptTemplate.QA_SYSTEM},
            {"role": "user", "content": f"Question: {question}"}
        ]

    @staticmethod
    def get_rag_messages(context: str, question: str):
        """Get messages for RAG task (利用 DeepSeek prompt caching)

        Args:
            context: Retrieved context
            question: Question to answer

        Returns:
            List of messages with system prompt (cached) and user message
        """
        user_message = f"""Context:
{context}

Question: {question}

Answer:"""

        return [
            {"role": "system", "content": PromptTemplate.RAG_SYSTEM},
            {"role": "user", "content": user_message}
        ]

    @staticmethod
    def get_iterative_eval_messages(question: str, answer: str, context: str = "None"):
        """Get messages for Iterative RAG evaluation

        Args:
            question: Original question
            answer: Generated answer
            context: Retrieved context (optional)

        Returns:
            List of messages for LLM evaluation
        """
        user_message = f"""Question: {question}

Answer: {answer}

Context: {context if context else "None"}"""

        return [
            {"role": "system", "content": PromptTemplate.ITERATIVE_EVAL_SYSTEM},
            {"role": "user", "content": user_message}
        ]

    
    # Evaluation Methods
    @staticmethod
    def get_answer_label_messages(question: str, ground_truth: str, predicted: str):
        """Get messages for classifying answer correctness

        Args:
            question: The question being asked
            ground_truth: Ground truth answer
            predicted: Predicted answer

        Returns:
            List of messages for LLM classification
        """
        user_message = f"""Question: {question}

Ground Truth Answer: {ground_truth}

Predicted Answer: {predicted}"""

        return [
            {"role": "system", "content": PromptTemplate.ANSWER_LABEL_SYSTEM},
            {"role": "user", "content": user_message}
        ]

    
    # Query Generation Prompts
    SINGLE_HOP_GEN_SYSTEM = """You are a dataset creator. Generate a simple factual question based on the passage.

### CRITICAL REQUIREMENT (Self-Containment):
The question must be understandable WITHOUT seeing the passage.
- **BAD:** "When was *he* born?" (Who is 'he'?)
- **GOOD:** "When was *Barack Obama* born?" (Specific Entity Name)
- **Action:** You MUST replace pronouns (he, she, it, they) with the actual entity names found in the text.

### Requirements:
1. The question should be answerable by a short phrase or sentence from the passage.
2. The answer must be explicitly stated in the passage.
3. Avoid yes/no questions.

Output JSON format:
{"question": "The self-contained question", "answer": "The extractable answer"}"""

    SUMMARY_GEN_SYSTEM = """You are an expert dataset creator.
Your task is to generate a **Synthesis Question** and a **Comprehensive Answer** based on a cluster of documents related to the entity: "{entity}".

### DOCUMENTS:
{documents}

### STEP-BY-STEP INSTRUCTIONS:
1. **Consistency Check:** Do these documents talk about the SAME "{entity}"?
   - If one doc is about "Michael Jordan (Basketball)" and another about "Michael Jordan (Professor)", return {{"discard": true}}.

2. **Information Synthesis (The Answer):**
   - Identify common themes across the documents (e.g., Career, Impact, Controversy, Evolution).
   - Write a high-quality summary (3-5 sentences) that **integrates** facts from AT LEAST 2 different documents.
   - **Constraint:** Do not simply list facts. Connect them (e.g., "While Doc A mentions X, Doc B clarifies that Y...").

3. **Question Formulation (The Query):**
   - Write a question that naturally elicits the summary you just wrote.
   - **Specific Theme:** Instead of "Summarize X", ask about a specific aspect like "How did X's career evolve?" or "What are the key contributions of X?".
   - **Self-Contained:** The question MUST contain the full name of "{entity}". Do NOT use pronouns like "he/she" or vague phrases like "the provided text".
     - BAD: "Summarize his achievements mentioned in the text."
     - GOOD: "What were the major scientific achievements of Albert Einstein?"

### OUTPUT FORMAT (JSON):
{{
  "reasoning": "Step 1: Themes identified are... Step 2: Documents used are Doc [x] and Doc [y]...",
  "question": "The self-contained synthesis question",
  "answer": "The synthesized answer covering multiple docs"
}}
OR
{{"discard": true}}
"""

    MULTIHOP_NHOP_GEN_SYSTEM = """Generate a strict {num_hops}-hop question based on the provided {num_docs}-document chain.

### DOCUMENT CHAIN:
{chain_description}

### GENERATION LOGIC (Reverse Substitution):
1. **Analyze:** Identify the Start Entity (Doc 1), Bridge Entities (Intermediate), and Final Answer (Last Doc).
2. **Recursive Masking:** Starting from the end, replace every Bridge Entity's name with a functional description derived **strictly** from the previous document.
   - *Example:* Instead of "Hawaii", use "the location where Jurassic Park was filmed".
3. **Assembly:** The final question must nest these descriptions and contain **ONLY** the specific name of the Start Entity.

### CRITICAL CONSTRAINTS:
1. **Forbidden:** NEVER use the actual name of any Bridge Entity in the question.
2. **Dependency:** The question must become unanswerable if any single document is removed.
3. **Answer Isolation:** The Final Answer must NOT appear in any document except the last one.
4. **Grammar:** Ensure natural phrasing for nested structures.

### OUTPUT JSON:
{{
  "reasoning": "Step 1: Identify Bridge [Name] and mask it as '[Description]'; Step 2: Identify Bridge [Name] and mask it as '[Description]'; Step 3: Verify strict dependency.",
  "question": "The final nested question",
  "answer": "The final answer from the last Doc"
}}"""

    
    # Query Generation Methods
    @staticmethod
    def get_single_hop_gen_messages(passage: str):
        """Get messages for single-hop query generation

        Args:
            passage: Source passage to generate question from

        Returns:
            List of messages for LLM query generation
        """
        user_message = f"""Passage:
{passage}

Generate a factual question and answer based on this passage."""

        return [
            {"role": "system", "content": PromptTemplate.SINGLE_HOP_GEN_SYSTEM},
            {"role": "user", "content": user_message}
        ]

    @staticmethod
    def get_summary_gen_messages(entity: str, documents: list):
        """Get messages for summary query generation

        Args:
            entity: Target entity for summarization
            documents: List of documents containing the entity

        Returns:
            List of messages for LLM query generation
        """
        # Format documents for the prompt
        docs_text = ""
        for i, doc in enumerate(documents, 1):
            doc_text = doc[:800] if len(doc) > 800 else doc
            docs_text += f"[Document {i}]:\n{doc_text}\n\n"

        # Fill variables in system prompt
        system_prompt = PromptTemplate.SUMMARY_GEN_SYSTEM.format(
            entity=entity,
            documents=docs_text.strip()
        )

        user_message = "Generate the synthesis question and answer based on the instructions above."

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

    @staticmethod
    def get_multihop_2hop_gen_messages(entity: str, doc1: str, doc2: str, title1: str = "", title2: str = ""):
        """Get messages for 2-hop multi-hop query generation (wrapper for backward compatibility)

        Args:
            entity: Bridge entity connecting both documents
            doc1: First document
            doc2: Second document
            title1: Title of first document (optional, ignored)
            title2: Title of second document (optional, ignored)

        Returns:
            List of messages for LLM query generation
        """
        # Delegate to the general N-hop function
        return PromptTemplate.get_multihop_nhop_gen_messages(
            documents=[doc1, doc2],
            bridges=[entity]
        )

    @staticmethod
    def get_multihop_nhop_gen_messages(documents: list, bridges: list):
        """Get messages for N-hop multi-hop query generation

        Args:
            documents: List of document texts [doc1, doc2, ..., docN]
            bridges: List of bridge entities [bridge_1, bridge_2, ...] (len = len(docs) - 1)

        Returns:
            List of messages for LLM query generation
        """
        num_docs = len(documents)
        num_hops = num_docs - 1

        # Build chain description dynamically
        chain_parts = []
        for i in range(num_docs):
            doc_text = documents[i][:500] if len(documents[i]) > 500 else documents[i]
            if i < num_docs - 1:
                chain_parts.append(f"[Doc {i+1}]: \"{doc_text}\"\n  --({bridges[i]})--> ")
            else:
                chain_parts.append(f"[Doc {i+1}]: \"{doc_text}\"")

        chain_description = "".join(chain_parts)

        system_prompt = PromptTemplate.MULTIHOP_NHOP_GEN_SYSTEM.format(
            num_hops=num_hops,
            num_docs=num_docs,
            chain_description=chain_description
        )

        user_message = f"Generate a {num_hops}-hop question based on the document chain above."

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
