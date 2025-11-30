import json
import requests
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class ReasoningAgent:
    def __init__(self, api_key, api_base, model):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.total_calls = 0
        self._lock = threading.Lock()
        self.session = requests.Session()  # Reuse connections
    
    def call_llm(self, prompt, max_tokens=4096, system_message=None, max_retries=3):
        """Call the LLM API with retry logic"""
        with self._lock:
            self.total_calls += 1
        
        # Truncate prompt if too long
        max_prompt_chars = 20000  # ~5000 tokens
        if len(prompt) > max_prompt_chars:
            prompt = prompt[:max_prompt_chars] + "\n\n[Input truncated due to length]"
        
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0
        }
        
        # Retry logic with exponential backoff
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.session.post(url, headers=headers, json=data, timeout=60)
                response.raise_for_status()
                result = response.json()
                return result['choices'][0]['message']['content']
            
            except requests.exceptions.ConnectionError as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = 2 ** attempt
                    print(f"  [Retry {attempt+1}/{max_retries}] Connection error, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"API call failed after {max_retries} retries: Connection reset")
            
            except requests.exceptions.Timeout as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  [Retry {attempt+1}/{max_retries}] Timeout, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"API call failed after {max_retries} retries: Timeout")
            
            except requests.exceptions.HTTPError as e:
                # Don't retry 400/401/404 - these are request errors, not network
                if e.response.status_code in [400, 401, 404]:
                    raise Exception(f"API call failed: {str(e)}")
                
                # Retry 500/502/503 - server errors
                if e.response.status_code >= 500:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"  [Retry {attempt+1}/{max_retries}] Server error {e.response.status_code}, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception(f"API call failed after {max_retries} retries: {str(e)}")
                
                # Other HTTP errors - don't retry
                raise Exception(f"API call failed: {str(e)}")
            
            except Exception as e:
                # Unexpected errors - raise immediately
                raise Exception(f"API call failed: {str(e)}")
        
        # Should never reach here
        raise Exception(f"API call failed after {max_retries} retries: {last_error}")
    
    def chain_of_thought(self, question, max_tokens=4096):
        """Chain-of-thought reasoning"""
        prompt = f"""Solve this problem step by step.

Problem: {question}

Think through this carefully, showing your reasoning. At the end, clearly state your final answer.

Therefore, the final answer is:"""
        
        return self.call_llm(prompt, max_tokens=max_tokens)
    
    def self_consistency(self, question, n_samples=3, max_tokens=3072):
        """Generate multiple solutions and take majority vote"""
        responses = []
        for i in range(n_samples):
            prompt = f"""Answer this question. Be direct and concise.

Question: {question}

Answer:"""
            try:
                response = self.call_llm(prompt, max_tokens=max_tokens)
                responses.append(response)
            except Exception as e:
                # If one sample fails, continue with others
                print(f"  Sample {i+1} failed: {str(e)[:50]}...")
                continue
        
        if not responses:
            raise Exception("All self-consistency samples failed")
        
        # Extract answers from each response
        answers = [self._extract_answer(r, question, 'common_sense') for r in responses]
        
        # Return majority vote
        if answers:
            answer_counts = Counter(answers)
            return answer_counts.most_common(1)[0][0]
        return responses[0]
    
    def _extract_answer_math(self, response, question):
        """Extract numerical answer from math response"""
        bad_patterns = [
            r'find\s+', r'\\frac', r'where\s+m\s+and\s+n',
            r'reduced\s+fraction', r'compute', r'calculate',
            r'determine', r'what\s+is', r'how\s+many', r'given\s+that'
        ]
        
        def is_problem_text(line):
            line_lower = line.lower()
            return any(re.search(pattern, line_lower) for pattern in bad_patterns)
        
        patterns = [
            r'Therefore,?\s+the\s+final\s+answer\s+is:?\s*(\d+)',
            r'Final\s+answer:?\s*(\d+)',
            r'\\boxed\{(\d+)\}',
            r'answer\s+is:?\s*(\d+)',
            r'=\s*(\d+)\s*$'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
            if matches:
                return matches[-1]
        
        lines = response.split('\n')
        for line in reversed(lines):
            if is_problem_text(line):
                continue
            match = re.search(r'(?<![0-9+\-*/=])\b(\d{1,6})\b(?![0-9+\-*/=])', line)
            if match:
                return match.group(1)
        
        all_numbers = re.findall(r'\b(\d{1,6})\b', response)
        for num in reversed(all_numbers):
            context_pattern = f'.{{0,50}}{re.escape(num)}.{{0,50}}'
            contexts = re.findall(context_pattern, response, re.DOTALL)
            if contexts and not any(is_problem_text(ctx) for ctx in contexts):
                return num
        
        return all_numbers[-1] if all_numbers else ""
    
    def _extract_answer_common_sense(self, response, question):
        """Extract text answer from common sense question"""
        if isinstance(response, bool):
            return response
        
        if isinstance(response, str):
            response_lower = response.lower().strip()
            
            if response_lower in ['true', 'yes']:
                return True
            elif response_lower in ['false', 'no']:
                return False
            
            if re.search(r'\b(true)\b', response_lower):
                return True
            elif re.search(r'\b(false)\b', response_lower):
                return False
            
            patterns = [
                r'[Aa]nswer:?\s*(.+?)(?:\n|$)',
                r'[Tt]he answer is:?\s*(.+?)(?:\n|$)',
                r'[Ff]inal answer:?\s*(.+?)(?:\n|$)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response)
                if match:
                    answer = match.group(1).strip()
                    answer_lower = answer.lower()
                    if answer_lower in ['true', 'yes']:
                        return True
                    elif answer_lower in ['false', 'no']:
                        return False
                    answer = re.sub(r'[.!?]$', '', answer)
                    answer = re.sub(r'^\**|\**$', '', answer)
                    return answer.strip()
            
            lines = [l.strip() for l in response.split('\n') if l.strip()]
            if lines:
                candidates = [l for l in lines[-3:] if len(l) < 200]
                if candidates:
                    answer = candidates[-1]
                    answer_lower = answer.lower()
                    if answer_lower in ['true', 'yes']:
                        return True
                    elif answer_lower in ['false', 'no']:
                        return False
                    answer = re.sub(r'[.!?]$', '', answer)
                    return answer.strip()
                return lines[-1]
            
            return response.strip()
        
        return str(response)
    
    def _extract_answer_future_prediction(self, response, question):
        """Extract answer from future prediction"""
        list_pattern = r'\[([^\]]+)\]'
        match = re.search(list_pattern, response)
        if match:
            list_content = match.group(0)
            return list_content
        
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        match = re.search(boxed_pattern, response)
        if match:
            answer = match.group(1)
            return f"['{answer}']"
        
        patterns = [
            r'[Aa]nswer:?\s*(.+?)(?:\n|$)',
            r'[Ff]inal answer:?\s*(.+?)(?:\n|$)',
            r'[Pp]rediction:?\s*(.+?)(?:\n|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                answer = match.group(1).strip()
                if not answer.startswith('['):
                    return f"['{answer}']"
                return answer
        
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        if lines:
            answer = lines[-1]
            if not answer.startswith('['):
                return f"['{answer}']"
            return answer
        
        return "['']"
    
    def _extract_answer_coding(self, response, question):
        """Extract ONLY function body from code response"""
        response = re.sub(r'```(?:python)?\s*\n?', '', response)
        response = re.sub(r'```\s*$', '', response)
        
        lines = response.split('\n')
        body_lines = []
        in_function = False
        
        for line in lines:
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                continue
            
            if line.strip().startswith('def '):
                in_function = True
                continue
            
            if in_function and (line.startswith('    ') or line.startswith('\t')):
                body_lines.append(line)
            elif in_function and line.strip():
                break
        
        if body_lines:
            return '\n'.join(body_lines)
        
        indented_lines = [l for l in lines if l.startswith(('    ', '\t')) and l.strip()]
        if indented_lines:
            return '\n'.join(indented_lines)
        
        return response.strip()
    
    def _extract_answer_planning(self, response, question):
        """Extract LISP-style action sequence"""
        lines = response.split('\n')
        action_lines = []
        
        for line in lines:
            line = line.strip()
            
            if any(phrase in line.lower() for phrase in [
                'here is', 'here are', 'the action sequence', 'following',
                'based on', 'to achieve', 'solution:', 'step'
            ]):
                continue
            
            line = re.sub(r'^\d+\.?\s*', '', line)
            
            if line.startswith('(') and ')' in line:
                match = re.search(r'\([^)]+\)', line)
                if match:
                    action_lines.append(match.group(0))
        
        if action_lines:
            return '\n'.join(action_lines)
        
        return ""
    
    def _extract_answer(self, response, question, domain):
        """Extract answer based on domain"""
        if domain == 'math':
            return self._extract_answer_math(response, question)
        elif domain == 'common_sense':
            return self._extract_answer_common_sense(response, question)
        elif domain == 'future_prediction':
            return self._extract_answer_future_prediction(response, question)
        elif domain == 'coding':
            return self._extract_answer_coding(response, question)
        elif domain == 'planning':
            return self._extract_answer_planning(response, question)
        else:
            return response.strip()
    
    def solve_math(self, question):
        """Solve math problem using chain-of-thought"""
        response = self.chain_of_thought(question)
        return self._extract_answer(response, question, 'math')
    
    def solve_common_sense(self, question):
        """Solve common sense problem"""
        question_lower = question.lower()
        if question_lower.startswith('is ') or question_lower.startswith('are ') or \
           question_lower.startswith('can ') or question_lower.startswith('does ') or \
           question_lower.startswith('do '):
            prompt = f"""{question}

Answer with True or False only.

Answer:"""
            response = self.call_llm(prompt, max_tokens=512)
            return self._extract_answer(response, question, 'common_sense')
        else:
            response = self.self_consistency(question, n_samples=3)
            return self._extract_answer(response, question, 'common_sense')
    
    def solve_coding(self, question):
        """Generate code solution"""
        prompt = f"""Write a Python function to solve this problem. Return ONLY the function body (the indented lines inside the function), NOT the function definition line or imports.

{question}

Function body (indented lines only):"""
        response = self.call_llm(prompt, max_tokens=2048)
        return self._extract_answer(response, question, 'coding')
    
    def solve_planning(self, question):
        """Generate planning action sequence"""
        prompt = f"""{question}

IMPORTANT: Provide the action sequence in LISP format ONLY. Each action must be on a new line in the format:
(action-name parameter1 parameter2 ...)

Do not include explanations, numbering, or natural language. Output only the actions in parentheses.

Actions:"""
        response = self.call_llm(prompt, max_tokens=2048)
        return self._extract_answer(response, question, 'planning')
    
    def solve_future_prediction(self, question):
        """Solve future prediction question"""
        prompt = f"""{question}

Answer:"""
        response = self.call_llm(prompt, max_tokens=1024)
        return self._extract_answer(response, question, 'future_prediction')
    
    def solve(self, question, domain):
        """Route to domain-specific solver"""
        if domain == 'math':
            return self.solve_math(question)
        elif domain == 'common_sense':
            return self.solve_common_sense(question)
        elif domain == 'coding':
            return self.solve_coding(question)
        elif domain == 'planning':
            return self.solve_planning(question)
        elif domain == 'future_prediction':
            return self.solve_future_prediction(question)
        else:
            return self.call_llm(question)


def normalize_and_compare(predicted, expected, domain):
    """Compare predicted and expected answers"""
    if isinstance(expected, bool):
        if isinstance(predicted, bool):
            return predicted == expected
        if isinstance(predicted, str):
            pred_lower = str(predicted).lower().strip()
            if pred_lower in ['true', 'yes', '1']:
                return expected == True
            elif pred_lower in ['false', 'no', '0']:
                return expected == False
        return False
    
    pred_str = str(predicted).strip()
    exp_str = str(expected).strip()
    pred_norm = pred_str.lower()
    exp_norm = exp_str.lower()
    
    if pred_norm == exp_norm:
        return True
    
    if domain == 'future_prediction':
        pred_compact = re.sub(r'\s+', '', pred_norm)
        exp_compact = re.sub(r'\s+', '', exp_norm)
        if pred_compact == exp_compact:
            return True
    
    if domain == 'coding':
        pred_compact = re.sub(r'\s+', '', pred_norm)
        exp_compact = re.sub(r'\s+', '', exp_norm)
        if pred_compact == exp_compact:
            return True
        
        if len(pred_compact) > 30 and len(exp_compact) > 30:
            min_len = min(len(pred_compact), len(exp_compact))
            max_len = max(len(pred_compact), len(exp_compact))
            
            if pred_compact in exp_compact or exp_compact in pred_compact:
                overlap = min_len
                similarity = overlap / max_len
                if similarity > 0.5:
                    return True
            
            pred_tokens = set(re.findall(r'[a-z_][a-z0-9_]*', pred_compact))
            exp_tokens = set(re.findall(r'[a-z_][a-z0-9_]*', exp_compact))
            
            if pred_tokens and exp_tokens:
                overlap = len(pred_tokens & exp_tokens)
                union = len(pred_tokens | exp_tokens)
                jaccard = overlap / union if union > 0 else 0
                
                if jaccard >= 0.5:
                    return True
    
    if domain == 'planning':
        pred_actions = [l.strip() for l in str(predicted).split('\n') if l.strip()]
        exp_actions = [l.strip() for l in str(expected).split('\n') if l.strip()]
        
        pred_actions_norm = [re.sub(r'\s+', ' ', a.lower()) for a in pred_actions]
        exp_actions_norm = [re.sub(r'\s+', ' ', a.lower()) for a in exp_actions]
        
        if pred_actions_norm == exp_actions_norm:
            return True
        
        def extract_action_name(action):
            match = re.match(r'\(([^\s)]+)', action)
            return match.group(1) if match else action
        
        pred_action_names = [extract_action_name(a) for a in pred_actions_norm]
        exp_action_names = [extract_action_name(a) for a in exp_actions_norm]
        
        if len(pred_action_names) == len(exp_action_names):
            matches = sum(1 for p, e in zip(pred_action_names, exp_action_names) if p == e)
            if matches / len(exp_action_names) >= 0.7:
                return True
    
    if domain in ['math']:
        try:
            pred_num = int(re.sub(r'[,\s]', '', str(predicted)))
            exp_num = int(re.sub(r'[,\s]', '', str(expected)))
            if pred_num == exp_num:
                return True
        except:
            pass
    
    pred_clean = re.sub(r'[^\w\s]', '', pred_norm)
    exp_clean = re.sub(r'[^\w\s]', '', exp_norm)
    
    if pred_clean == exp_clean:
        return True
    
    if domain in ['common_sense']:
        if exp_clean in pred_clean or pred_clean in exp_clean:
            if len(exp_clean) > 10:
                return True
    
    return False


def process_single_item(agent, item):
    """Process a single item"""
    try:
        prediction = agent.solve(item['input'], item['domain'])
        success = normalize_and_compare(prediction, item['output'], item['domain'])
        
        return {
            'index': item['index'],
            'domain': item['domain'],
            'input': item['input'],
            'expected': item['output'],
            'predicted': prediction,
            'success': success,
            'error': None
        }
    except Exception as e:
        return {
            'index': item['index'],
            'domain': item['domain'],
            'input': item['input'],
            'expected': item['output'],
            'predicted': '',
            'success': False,
            'error': str(e)
        }


def process_batch_concurrent(agent, data, max_workers=5):
    """Process batch of examples using concurrent execution"""
    results = [None] * len(data)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(process_single_item, agent, item): item['index']
            for item in data
        }
        
        completed = 0
        for future in as_completed(future_to_index):
            result = future.result()
            results[result['index']] = result
            completed += 1
            
            if completed % 50 == 0:
                print(f"Progress: {completed}/{len(data)} completed")
    
    return results


def evaluate_results(results):
    """Evaluate and print statistics"""
    total = len(results)
    correct = sum(1 for r in results if r['success'])
    errors = sum(1 for r in results if r['error'] is not None)
    
    print(f"\n{'='*60}")
    print(f"OVERALL RESULTS")
    print(f"{'='*60}")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct - errors}")
    print(f"Errors: {errors}")
    print(f"Accuracy: {correct/total*100:.2f}%")
    
    from collections import defaultdict
    domain_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'errors': 0, 'failures': []})
    
    for r in results:
        domain = r['domain']
        domain_stats[domain]['total'] += 1
        if r['success']:
            domain_stats[domain]['correct'] += 1
        elif r['error'] is None:
            domain_stats[domain]['failures'].append((r['index'], r['expected'], r['predicted']))
        if r['error'] is not None:
            domain_stats[domain]['errors'] += 1
    
    print(f"\n{'='*60}")
    print(f"DOMAIN BREAKDOWN")
    print(f"{'='*60}")
    for domain in sorted(domain_stats.keys()):
        stats = domain_stats[domain]
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"{domain:20s}: {stats['correct']:3d}/{stats['total']:3d} = {acc:5.2f}% (errors: {stats['errors']})")
        
        if stats['failures'][:3]:
            print(f"  Example failures:")
            for idx, exp, pred in stats['failures'][:3]:
                exp_short = str(exp)[:50] + '...' if len(str(exp)) > 50 else str(exp)
                pred_short = str(pred)[:50] + '...' if len(str(pred)) > 50 else str(pred)
                print(f"    [{idx}] Expected: '{exp_short}' Got: '{pred_short}'")


def main():
    API_KEY = "cse476"
    API_BASE = "http://10.4.58.53:41701/v1"
    MODEL = "bens_model"
    
    print("Initializing agent with retry logic...")
    agent = ReasoningAgent(API_KEY, API_BASE, MODEL)
    
    print("Loading data...")
    with open('./cse476_final_project_dev_data.json', 'r') as f:
        dev_data = json.load(f)
    
    for i, item in enumerate(dev_data):
        item['index'] = i
    
    print(f"Loaded {len(dev_data)} examples")
    print(f"Processing with {5} concurrent workers and 3 retries per call...")
    
    results = process_batch_concurrent(agent, dev_data, max_workers=5)
    
    output_file = 'predictions_with_retries.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"Total API calls made: {agent.total_calls}")
    print(f"Average calls per question: {agent.total_calls/len(dev_data):.2f}")
    
    evaluate_results(results)


if __name__ == "__main__":
    main()