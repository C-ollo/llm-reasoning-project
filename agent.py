import os
import json
import time
import re
from typing import Dict, List, Any, Optional
from collections import Counter


import os
import json
import time
import re
from typing import Dict, List, Any, Optional
from collections import Counter
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class ReasoningAgent:
    """Agent that uses multiple inference-time techniques to solve reasoning problems."""
    
    def __init__(self, api_key: str = "cse476", 
                 api_base: str = "http://10.4.58.53:41701/v1",
                 model: str = "bens_model"):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.call_count = 0
        self._lock = threading.Lock()  # Thread safety for call_count
        
    def call_llm(self, prompt: str, system: str = None, 
                 temperature: float = 0.0, max_tokens: int = 2048) -> Optional[str]:
        """Call the LLM API and return the response text."""
        with self._lock:
            self.call_count += 1
        
        url = f"{self.api_base}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                print(f"API Error: {resp.status_code}")
                return None
        except Exception as e:
            print(f"Exception calling LLM: {e}")
            return None
        
    # ========================
    # TECHNIQUE 1: Chain-of-Thought Reasoning
    # ========================
    
    def chain_of_thought(self, question: str, domain: str) -> str:
        """Generate answer using step-by-step reasoning."""
        system = "You are a careful problem solver. Think step-by-step and show your reasoning."
        
        prompt = f"""{question}

Please solve this step-by-step:
1. Understand the problem
2. Plan your approach
3. Work through the solution
4. State the final answer clearly

End with: "Therefore, the answer is: [your answer]"
"""
        
        response = self.call_llm(prompt, system=system, max_tokens=2048)
        if not response:
            return ""
        
        # Extract final answer
        answer = self._extract_answer(response, domain)
        return answer

    def self_consistency(self, question: str, domain: str, n_samples: int = 3) -> str:
        """Generate multiple solutions and use majority voting."""
        answers = []
        
        system = "You are a problem solver. Provide a clear final answer."
        
        for i in range(n_samples):
            prompt = f"""{question}

Solve this problem and provide your final answer clearly.
For the final answer, use the format: "Final answer: [your answer]"
"""
            # Use different temperatures for diversity
            temp = 0.3 if i > 0 else 0.0
            response = self.call_llm(prompt, system=system, temperature=temp, max_tokens=2048)
            
            if response:
                answer = self._extract_answer(response, domain)
                if answer:
                    answers.append(answer)
        
        # Majority voting
        if not answers:
            return ""
        
        # For coding/planning, return the longest/most complete answer
        if domain in ['coding', 'planning']:
            return max(answers, key=len)
        
        # For others, use most common answer
        counter = Counter(answers)
        return counter.most_common(1)[0][0]
    
    # ========================
    # TECHNIQUE 3: Self-Verification
    # ========================
    
    def self_verify(self, question: str, initial_answer: str, domain: str) -> str:
        """Verify and potentially correct the initial answer."""
        system = "You are a critical reviewer. Check if the answer is correct."
        
        prompt = f"""Question: {question}

Proposed Answer: {initial_answer}

Verify this answer:
1. Is the reasoning correct?
2. Are there any errors?
3. What is the correct answer?

Provide the verified final answer in the format: "Verified answer: [answer]"
"""
        
        response = self.call_llm(prompt, system=system, max_tokens=2048)
        if not response:
            return initial_answer
        
        verified = self._extract_answer(response, domain)
        return verified if verified else initial_answer
        