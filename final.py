#!/usr/bin/env python3
"""
CONCURRENT FULL RUN SCRIPT - Process all 6208 test questions with 5 concurrent workers
Run this AFTER test_sample.py succeeds!

CONCURRENT VERSION: ~1-1.5 hours instead of 5-6 hours!

Usage: python3 generate_final_answers.py
"""

import json
import requests
import re
import time
import threading
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# File paths
INPUT_PATH = Path("./cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("./cse_476_final_project_answers.json")

# API Configuration
API_KEY = "cse476"
API_BASE = "http://10.4.58.53:41701/v1"
MODEL = "bens_model"


class ReasoningAgent:
    """Inference-time reasoning agent with retry logic and thread-safe operation"""
    
    def __init__(self, api_key, api_base, model):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.total_calls = 0
        self._lock = threading.Lock()  # Thread-safe counter
        self.session = requests.Session()  # Connection pooling
    
    def call_llm(self, prompt, max_tokens=4096, max_retries=3):
        """Call LLM API with retry logic (thread-safe)"""
        with self._lock:
            self.total_calls += 1
        
        # Truncate if too long (model limit is 8192 tokens)
        max_prompt_chars = 20000  # ~5000 tokens
        if len(prompt) > max_prompt_chars:
            prompt = prompt[:max_prompt_chars] + "\n\n[Input truncated due to length]"
        
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0
        }
        
        # Retry with exponential backoff (1s, 2s, 4s)
        for attempt in range(max_retries):
            try:
                response = self.session.post(url, headers=headers, json=data, timeout=200)
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content']
            
            except requests.exceptions.ConnectionError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                raise Exception(f"Connection failed after {max_retries} retries")
            
            except requests.exceptions.Timeout as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                raise Exception(f"Timeout after {max_retries} retries")
            
            except requests.exceptions.HTTPError as e:
                # Don't retry client errors (400, 401, 404)
                if e.response.status_code in [400, 401, 404]:
                    raise Exception(f"HTTP {e.response.status_code}: {str(e)}")
                if e.response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                        continue
                    raise Exception(f"Rate limit exceeded after {max_retries} retries")
                
                # Retry server errors (500, 502, 503)
                if e.response.status_code >= 500:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                        continue
                
                raise Exception(f"HTTP error: {str(e)}")
            
            except Exception as e:
                raise Exception(f"API call failed: {str(e)}")
        
        raise Exception(f"Failed after {max_retries} retries")
    
    def solve(self, question):
        """Solve question using Chain-of-Thought reasoning"""
        prompt = f"""Answer this question step by step. Think carefully and provide your reasoning but keep it concise.

Question: {question}

After explaining your reasoning concisely, state your final answer clearly.

Final answer:"""
        
        try:
            response = self.call_llm(prompt)
            return self._extract_answer(response, question)
        except Exception as e:
            # Return error message instead of crashing
            return f"Error: {str(e)[:100]}"
    
    def _extract_answer(self, response, question):
        """Extract clean answer from response"""
        
        # Multiple choice (A, B, C, D, E, etc.)
        if re.search(r'\b[A-E]\)', question) or re.search(r'\b[A-E]\.', question):
            patterns = [
                r'[Ff]inal answer:?\s*([A-E])',
                r'[Aa]nswer:?\s*([A-E])',
                r'[Tt]he answer is:?\s*([A-E])',
                r'\b([A-E])\b\s*(?:is\s+(?:the\s+)?(?:correct|right))',
                r'[Cc]hoose\s+([A-E])',
                r'\b([A-E])\b(?=\s*$)',  # Single letter at end
            ]
            for pattern in patterns:
                match = re.search(pattern, response)
                if match:
                    return match.group(1)
        
        # True/False questions
        if 'true or false' in question.lower() or 'is it true' in question.lower():
            if re.search(r'\b(true)\b', response.lower()):
                return "True"
            elif re.search(r'\b(false)\b', response.lower()):
                return "False"
        
        # Numbers (math/word problems)
        number_patterns = [
            r'[Ff]inal answer:?\s*\$?(\d+(?:,\d{3})*(?:\.\d+)?)',
            r'[Aa]nswer:?\s*\$?(\d+(?:,\d{3})*(?:\.\d+)?)',
            r'[Tt]he answer is:?\s*\$?(\d+(?:,\d{3})*(?:\.\d+)?)',
            r'\\boxed\{(\d+(?:,\d{3})*(?:\.\d+)?)\}',
        ]
        for pattern in number_patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).replace(',', '')
        
        # Text answers - extract last substantive line
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        if lines:
            for line in reversed(lines):
                if any(marker in line.lower() for marker in ['final answer', 'answer:', 'therefore', 'thus,']):
                    clean = re.sub(r'^[Ff]inal answer:?\s*', '', line)
                    clean = re.sub(r'^[Aa]nswer:?\s*', '', clean)
                    clean = re.sub(r'^[Tt]herefore,?\s*', '', clean)
                    clean = re.sub(r'^[Tt]hus,?\s*', '', clean)
                    clean = re.sub(r'[.!?]$', '', clean)
                    clean = clean.strip()
                    if clean and len(clean) < 500:
                        return clean
            
            for line in reversed(lines):
                if 20 < len(line) < 500:
                    return line
            
            if lines[-1]:
                return lines[-1]
        
        return response[:500].strip()


def process_single_question(agent: ReasoningAgent, question: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Process a single question (for concurrent execution)"""
    try:
        answer = agent.solve(question["input"])
        
        # Ensure it's a string
        answer_str = str(answer)
        
        # Truncate if too long (autograder limit is 5000 chars)
        if len(answer_str) >= 5000:
            answer_str = answer_str[:4900] + "..."
        
        return {
            "index": index,
            "output": answer_str,
            "success": True
        }
    
    except Exception as e:
        # If processing fails completely, return error
        return {
            "index": index,
            "output": f"Processing error: {str(e)[:100]}",
            "success": False
        }


def process_concurrent(agent: ReasoningAgent, questions: List[Dict[str, Any]], max_workers: int = 50) -> List[Dict[str, str]]:
    """Process questions concurrently using ThreadPoolExecutor"""
    
    total = len(questions)
    results = [None] * total  # Pre-allocate results array
    
    print(f"Processing {total} questions with {max_workers} concurrent workers...")
    print(f"Expected runtime: ~{total / max_workers * 3.5 / 3600:.1f} hours")
    print("=" * 80)
    
    start_time = time.time()
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_single_question, agent, q, idx): idx
            for idx, q in enumerate(questions)
        }
        
        # Process as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            result = future.result()
            results[index] = result
            
            completed += 1
            
            # Progress updates every 100 completions
            if completed % 100 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed  # questions per second
                remaining = (total - completed) / rate if rate > 0 else 0
                print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%) | "
                      f"Elapsed: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m | "
                      f"Rate: {rate*60:.1f} q/min")
    
    elapsed = time.time() - start_time
    
    print()
    print(f"Completed! Processed {total} questions in {elapsed/60:.1f} minutes")
    print(f"Average rate: {total/elapsed*60:.1f} questions/minute")
    
    # Extract just the outputs in order
    answers = [{"output": r["output"]} for r in results]
    
    # Count errors
    errors = sum(1 for r in results if not r.get("success", True))
    if errors > 0:
        print(f"⚠️  Warning: {errors} questions had errors")
    
    return answers


def load_questions(path: Path) -> List[Dict[str, Any]]:
    """Load test questions from JSON file"""
    with path.open("r") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Input file must contain a list of question objects.")
    return data


def validate_results(questions: List[Dict[str, Any]], answers: List[Dict[str, Any]]) -> None:
    """Validate answer format for autograder"""
    if len(questions) != len(answers):
        raise ValueError(f"Mismatched lengths: {len(questions)} questions vs {len(answers)} answers.")
    
    for idx, answer in enumerate(answers):
        if "output" not in answer:
            raise ValueError(f"Missing 'output' field for answer index {idx}.")
        if not isinstance(answer["output"], str):
            raise TypeError(f"Answer at index {idx} has non-string output: {type(answer['output'])}")
        if len(answer["output"]) >= 5000:
            raise ValueError(f"Answer at index {idx} exceeds 5000 characters ({len(answer['output'])} chars).")


def main() -> None:
    """Main execution"""
    print()
    print("=" * 80)
    print("CSE476 FINAL PROJECT - TEST DATA PROCESSING (CONCURRENT)")
    print("=" * 80)
    print()
    print("🚀 CONCURRENT VERSION - Using 5 workers!")
    print("🚀 Expected runtime: ~1-1.5 hours (instead of 5-6 hours!)")
    print()
    print("⚠️  This will process 6,208 questions!")
    print("⚠️  Make sure:")
    print("   - You're on ASU network/VPN")
    print("   - Computer won't sleep")
    print("   - You have stable connection")
    print()
    print("Press Ctrl+C now to cancel, or wait 5 seconds to continue...")
    print()
    
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        return
    
    print("Starting concurrent processing...")
    print()
    
    # Initialize agent
    print("Initializing reasoning agent with retry logic...")
    agent = ReasoningAgent(API_KEY, API_BASE, MODEL)
    print("✅ Agent initialized")
    print()
    
    # Load questions
    print(f"Loading test data from {INPUT_PATH}...")
    questions = load_questions(INPUT_PATH)
    print(f"✅ Loaded {len(questions)} test questions")
    print()
    
    # Process concurrently
    answers = process_concurrent(agent, questions, max_workers=20)
    
    # Save results
    print()
    print(f"Saving answers to {OUTPUT_PATH}...")
    with OUTPUT_PATH.open("w") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)
    print(f"✅ Saved to {OUTPUT_PATH}")
    
    # Validate
    print()
    print("Validating format...")
    with OUTPUT_PATH.open("r") as fp:
        saved_answers = json.load(fp)
    validate_results(questions, saved_answers)
    print("✅ Format validated successfully")
    
    # Summary
    print()
    print("=" * 80)
    print("✅ SUCCESS! CONCURRENT PROCESSING COMPLETE!")
    print("=" * 80)
    print()
    print(f"✅ Processed: {len(answers)} questions")
    print(f"✅ Output file: {OUTPUT_PATH}")
    print(f"✅ Total API calls: {agent.total_calls}")
    print(f"✅ Average calls/question: {agent.total_calls/len(questions):.2f}")
    print()
    print("=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print()
    print("1. Verify the output file looks correct:")
    print(f"   head -50 {OUTPUT_PATH}")
    print()
    print("2. Check file size (should be ~2-3 MB):")
    print(f"   ls -lh {OUTPUT_PATH}")
    print()
    print("3. Submit to autograder:")
    print(f"   - File: {OUTPUT_PATH}")
    print(f"   - Code: generate_final_answers.py")
    print(f"   - Report: REPORT.md")
    print(f"   - GitHub: [your repository]")
    print()
    print("🎉 Congratulations on completing your CSE476 final project! 🎓✨")
    print()


if __name__ == "__main__":
    main()