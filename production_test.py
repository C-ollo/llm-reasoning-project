#!/usr/bin/env python3
"""
CSE476 Final Project - Concurrent Test Data Processing
Processes 6208 test questions using concurrent workers with retry logic
"""

import json
import requests
import re
import time
import threading
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

INPUT_PATH = Path("./cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("./cse_476_final_project_answers.json")

API_KEY = "cse476"
API_BASE = "http://10.4.58.53:41701/v1"
MODEL = "bens_model"


class ReasoningAgent:
    
    def __init__(self, api_key, api_base, model):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.total_calls = 0
        self._lock = threading.Lock()
        self.session = requests.Session()
    
    def call_llm(self, prompt, max_tokens=4096, max_retries=3):
        with self._lock:
            self.total_calls += 1
        
        max_prompt_chars = 20000
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
                if e.response.status_code in [400, 401, 404]:
                    raise Exception(f"HTTP {e.response.status_code}: {str(e)}")
                if e.response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                        continue
                    raise Exception(f"Rate limit exceeded after {max_retries} retries")
                
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
        prompt = f"""Answer this question step by step. Think carefully and provide your reasoning but keep it concise.

Question: {question}

After explaining your reasoning concisely, state your final answer clearly.

Final answer:"""
        
        try:
            response = self.call_llm(prompt)
            return self._extract_answer(response, question)
        except Exception as e:
            return f"Error: {str(e)[:100]}"
    
    def _extract_answer(self, response, question):
        
        if re.search(r'\b[A-E]\)', question) or re.search(r'\b[A-E]\.', question):
            patterns = [
                r'[Ff]inal answer:?\s*([A-E])',
                r'[Aa]nswer:?\s*([A-E])',
                r'[Tt]he answer is:?\s*([A-E])',
                r'\b([A-E])\b\s*(?:is\s+(?:the\s+)?(?:correct|right))',
                r'[Cc]hoose\s+([A-E])',
                r'\b([A-E])\b(?=\s*$)',
            ]
            for pattern in patterns:
                match = re.search(pattern, response)
                if match:
                    return match.group(1)
        
        if 'true or false' in question.lower() or 'is it true' in question.lower():
            if re.search(r'\b(true)\b', response.lower()):
                return "True"
            elif re.search(r'\b(false)\b', response.lower()):
                return "False"
        
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
    try:
        answer = agent.solve(question["input"])
        answer_str = str(answer)
        
        if len(answer_str) >= 5000:
            answer_str = answer_str[:4900] + "..."
        
        return {
            "index": index,
            "output": answer_str,
            "success": True
        }
    
    except Exception as e:
        return {
            "index": index,
            "output": f"Processing error: {str(e)[:100]}",
            "success": False
        }


def process_concurrent(agent: ReasoningAgent, questions: List[Dict[str, Any]], max_workers: int = 50) -> List[Dict[str, str]]:
    total = len(questions)
    results = [None] * total
    
    print(f"Processing {total} questions with {max_workers} concurrent workers...")
    
    start_time = time.time()
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(process_single_question, agent, q, idx): idx
            for idx, q in enumerate(questions)
        }
        
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            result = future.result()
            results[index] = result
            
            completed += 1
            
            if completed % 100 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                remaining = (total - completed) / rate if rate > 0 else 0
                print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%) | "
                      f"Elapsed: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m")
    
    elapsed = time.time() - start_time
    print(f"\nCompleted {total} questions in {elapsed/60:.1f} minutes")
    
    answers = [{"output": r["output"]} for r in results]
    
    errors = sum(1 for r in results if not r.get("success", True))
    if errors > 0:
        print(f"Warning: {errors} questions had errors")
    
    return answers


def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Input file must contain a list of question objects.")
    return data


def validate_results(questions: List[Dict[str, Any]], answers: List[Dict[str, Any]]) -> None:
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
    print(f"Loading test data from {INPUT_PATH}...")
    questions = load_questions(INPUT_PATH)
    print(f"Loaded {len(questions)} test questions\n")
    
    agent = ReasoningAgent(API_KEY, API_BASE, MODEL)
    
    answers = process_concurrent(agent, questions, max_workers=20)
    
    print(f"\nSaving answers to {OUTPUT_PATH}...")
    with OUTPUT_PATH.open("w") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)
    
    with OUTPUT_PATH.open("r") as fp:
        saved_answers = json.load(fp)
    validate_results(questions, saved_answers)
    
    print(f"\nProcessing complete!")
    print(f"Total questions: {len(answers)}")
    print(f"Total API calls: {agent.total_calls}")
    print(f"Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()