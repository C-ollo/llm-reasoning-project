# Inference-Time Reasoning Agent

---

## Overview

This project implements an inference-time reasoning agent that uses Chain-of-Thought (CoT), Self-Consistency, and domain-specific prompting to solve problems across 5 domains. 
**Key Features:**
- Multiple reasoning techniques (CoT, Self-Consistency, Domain-Specific)
- Concurrent processing with 5 workers (5x speedup)
- Robust error handling with retry logic
- Production-quality thread-safe operations

---

## Quick Start

### Prerequisites

```bash
# Python 3.8+ required
python3 --version


**Dependencies:**
- `requests` - HTTP client for API calls
- `json`, `re`, `time`, `threading`, `pathlib`, `concurrent.futures` - Standard library

---


### 2. Test Data (6,208 examples)

**File:** `generate_final_answers.py`

**What it does:** Processes 6,208 test examples using concurrent processing with 5 workers

**How to run:**
```bash
python3 generate_final_answers.py
```

- **429 backoff:** 3s, 6s, 12s, 24s, 48s (longer for rate limits)


## Reasoning Techniques

### 1. Chain-of-Thought (CoT)

**Used for:** Math domain (dev) and all test data

**How it works:**
```python
prompt = f"""Solve this problem step by step.

Problem: {question}

Think through this carefully, showing your reasoning. 
At the end, clearly state your final answer.

Therefore, the final answer is:"""
```

**Why:** Enables step-by-step reasoning for complex problems

**Results:** 27.67% on 300 AIME-level math problems

---

### 2. Self-Consistency with Majority Voting

**Used for:** Common sense domain (dev)

**How it works:**
- Generates 3 independent answers
- Extracts answer from each response
- Takes majority vote
- Returns most common answer

**Why:** Reduces random errors through consensus

**Results:** 30.00% on 400 common sense questions

**API calls:** 3 per question (within <20 constraint)

---

### 3. Domain-Specific Prompting

**Used for:** All 5 domains in dev data

**Domains:**
- **Math:** Step-by-step reasoning, numerical extraction
- **Common Sense:** True/False detection, boolean handling
- **Coding:** Function body extraction, token-based comparison
- **Planning:** LISP format parsing
- **Future Prediction:** List format extraction

**Why:** Each domain has unique requirements and answer formats

---

## Concurrent Processing Details

### Why Concurrent?

**Sequential processing:**
- 1 question at a time
- 6,208 questions × 3-4 seconds = ~5-6 hours ❌

**Concurrent processing (5 workers):**
- 5 questions simultaneously
- 6,208 questions ÷ 5 × 3-4 seconds = ~1 hour ✅
- **5x speedup!**

### Thread Safety

```python
class ReasoningAgent:
    def __init__(self):
        self._lock = threading.Lock()  # Protects shared state
        self.total_calls = 0
    
    def call_llm(self):
        with self._lock:
            self.total_calls += 1  # Thread-safe increment
```

**Why important:** Multiple threads accessing shared variables can cause race conditions

---

## Error Handling

### Retry Logic

**Network errors (Connection, Timeout):**
- Retry 3-5 times with exponential backoff
- Wait: 1s, 2s, 4s, 8s, 16s
- Prevents transient failures

**Rate limit errors (429):**
- Special longer backoff: 3s, 6s, 12s, 24s, 48s
- Respects server rate limits
- Critical for concurrent processing

**Example:**
```python
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 429:
        wait_time = 3 * (2 ** attempt)  # Longer wait
        time.sleep(wait_time)
        continue
```


## Implementation Highlights

### Production Quality Features

✅ **Thread-safe concurrent processing** - 5 workers, no race conditions  
✅ **Comprehensive error handling** - Retry logic, exponential backoff  
✅ **Rate limit management** - Special 429 handling, respects server limits  
✅ **Connection pooling** - Reuses TCP connections for stability  
✅ **Input validation** - Truncates long inputs, handles edge cases  
✅ **Progress tracking** - Real-time ETA and completion stats  
✅ **Format validation** - Ensures output meets autograder requirements  

### Code Quality

✅ **Clean architecture** - Modular design with ReasoningAgent class  
✅ **Well-documented** - Inline comments, clear variable names  
✅ **Robust extraction** - Multi-pattern matching with fallbacks  
✅ **Type safety** - Handles boolean/string/numeric responses  
✅ **Graceful degradation** - Continues on non-fatal errors  

---


## Quick Reference Commands

```bash
# Run dev data processing
python3 reasoning_agent.py

