#!/usr/bin/env python3
"""
IMCI Graph Testing Harness - Local Inference Script

This script runs inference on generated MCQA data using various AI models via AI Suite.
Supports multiple model families across different providers
"""

import json
import time
import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import aisuite as ai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for a model to test."""
    name: str
    provider: str
    model_id: str

# Define models to test with AI Suite
MODELS = [
    # OpenAI Models
    ModelConfig("o4-mini", "openai", "o4-mini-2025-04-16"),
    ModelConfig("gpt-oss-20b", "ollama", "gpt-oss:20b"),
    # Local Models (via Ollama through AI Suite)
    ModelConfig("medgemma-4b", "ollama", "alibayram/medgemma:4b"),
    ModelConfig("olmo2-7b", "ollama", "olmo2:7b"),
    ModelConfig("qwen3-1.7b", "ollama", "qwen3:1.7b"),
    ModelConfig("llama3.1-8b", "ollama", "llama3.1:8b"),
    ModelConfig("phi3-3.8b", "ollama", "phi3:3.8b"),
]

class AISuiteInference:
    """Handles inference using AI Suite."""
    
    def __init__(self):
        self.client = ai.Client()
        logger.info("AI Suite Client initialized successfully")
    
    def check_model_available(self, provider: str, model_id: str) -> bool:
        """Check if a model is available."""
        try:
            # For local models (Ollama), we'll check differently
            if provider == "ollama":
                return self._check_ollama_model(model_id)
            
            # For cloud models, we'll try a simple test
            return self._test_cloud_model(provider, model_id)
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False
    
    def _check_ollama_model(self, model_id: str) -> bool:
        """Check if an Ollama model is available."""
        try:
            import subprocess
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if parts and parts[0] == model_id:
                        return True
            return False
        except Exception:
            return False
    
    def _test_cloud_model(self, provider: str, model_id: str) -> bool:
        """Test if a cloud model is available by checking API keys."""
        if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            logger.warning("OpenAI API key not found")
            return False
        
        return True
    
    def generate_response(self, provider: str, model_id: str, prompt: str, max_tokens: int = 100, max_retries: int = 3) -> Optional[str]:
        """Generate a response using AI Suite with timeout and retry logic."""
        import time
        
        # Set model-specific temperature based on official documentation and research
        def get_optimal_temperature(provider: str, model_id: str) -> float:
            if provider == "openai" and model_id.startswith("o"):  # o3, o4-mini models
                return None
            return 0.0
            
        for attempt in range(max_retries):
            try:
                model_name = f"{provider}:{model_id}"
                messages = [{"role": "user", "content": prompt}]
                timeout = 30  # Default 30 seconds
                temperature = get_optimal_temperature(provider, model_id)
                
                # Handle different parameters for different models
                if provider == "openai" and model_id.startswith("o"):
                    # o4-mini reasoning models don't support temperature parameter
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        timeout=timeout
                    )
                else:
                    # All other models support temperature parameter
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        timeout=timeout
                    )
                
                return response.choices[0].message.content.strip() if response and response.choices else None
                
            except Exception as e:
                error_msg = str(e)
                is_timeout = "timed out" in error_msg.lower() or "timeout" in error_msg.lower()
                
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 30  # Progressive backoff: 30s, 60s, 90s
                    if is_timeout:
                        logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries}, retrying in {wait_time}s: {e}")
                    else:
                        logger.warning(f"Error on attempt {attempt + 1}/{max_retries}, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    if is_timeout:
                        logger.error(f"Final timeout after {max_retries} attempts: {e}")
                    else:
                        logger.error(f"Final error after {max_retries} attempts: {e}")
                    return None
    


class MCQAEvaluator:
    """Evaluates multiple choice questions using AI models."""
    
    def __init__(self, ai_suite: AISuiteInference):
        self.ai_suite = ai_suite
    
    def format_question_prompt(self, question_data: Dict[str, Any]) -> str:
        """Format a question for the model prompt."""
        question = question_data["question"]
        options = question_data["options"]
        
        prompt = f"""You are a medical expert answering questions about childhood illness management.

Question: {question}

Options:
A) {options['A']}
B) {options['B']}
C) {options['C']}
D) {options['D']}

Please answer with only the letter (A, B, C, or D) that corresponds to the correct answer.

Answer:"""
        
        return prompt
    
    def extract_answer(self, response: str) -> str:
        """Extract the answer letter from the model response."""
        # Clean the response and look for A, B, C, or D
        response = response.strip().upper()
        
        # Look for single letter answers
        for char in response:
            if char in ['A', 'B', 'C', 'D']:
                return char
        
        # If no single letter found, look for patterns like "The answer is A" or "A is correct"
        import re
        patterns = [
            r'answer.*?([ABCD])',
            r'([ABCD]).*?correct',
            r'([ABCD]).*?right',
            r'option.*?([ABCD])',
            r'choice.*?([ABCD])'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # Default to first option if no clear answer found
        return 'A'
    
    def evaluate_question(self, provider: str, model_id: str, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single question with a model."""
        prompt = self.format_question_prompt(question_data)
        
        start_time = time.time()
        response = self.ai_suite.generate_response(provider, model_id, prompt)
        end_time = time.time()
        
        if response is None:
            return {
                "provider": provider,
                "model_id": model_id,
                "question_id": question_data["id"],
                "correct_answer": question_data["correct_answer"],
                "predicted_answer": None,
                "is_correct": False,
                "response": None,
                "response_time": None,
                "error": "Failed to generate response"
            }
        
        predicted_answer = self.extract_answer(response)
        is_correct = predicted_answer == question_data["correct_answer"]
        
        return {
            "provider": provider,
            "model_id": model_id,
            "question_id": question_data["id"],
            "correct_answer": question_data["correct_answer"],
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "response": response,
            "response_time": end_time - start_time,
            "error": None
        }

def load_questions(data_path: str) -> List[Dict[str, Any]]:
    """Load questions from JSON file."""
    try:
        with open(data_path, 'r') as f:
            questions = json.load(f)
        logger.info(f"Loaded {len(questions)} questions from {data_path}")
        return questions
    except Exception as e:
        logger.error(f"Error loading questions: {e}")
        return []

def get_completed_records_from_jsonl(jsonl_path):
    """Return a set of (model_name, question_id) for records already in the JSONL file."""
    completed = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    model_name = rec.get('model_name')
                    question_id = rec.get('question_id')
                    if model_name is not None and question_id is not None:
                        completed.add((model_name, question_id))
                except Exception:
                    continue
    return completed

def run_inference(data_path: str, models: List[str], max_questions: Optional[int] = None, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Run inference on the dataset with specified models, skipping (model_name, question_id) pairs already in JSONL."""
    # Load questions
    questions = load_questions(data_path)
    if not questions:
        return {"error": "No questions loaded"}
    
    # Limit questions if specified
    if max_questions:
        questions = questions[:max_questions]
        logger.info(f"Limited to {max_questions} questions for testing")
    
    # Prepare JSONL path and completed records set
    jsonl_path = None
    completed_records = set()
    if output_dir:
        jsonl_path = os.path.join(output_dir, "inference_results.jsonl")
        completed_records = get_completed_records_from_jsonl(jsonl_path)
    
    # Initialize AI Suite
    ai_suite = AISuiteInference()
    evaluator = MCQAEvaluator(ai_suite)
    
    # Filter available models
    available_models = []
    for model in MODELS:
        if model.name in models and ai_suite.check_model_available(model.provider, model.model_id):
            available_models.append(model)
            logger.info(f"Model {model.name} ({model.provider}) is available")
        elif model.name in models:
            logger.warning(f"Model {model.name} ({model.provider}) is not available")
    
    if not available_models:
        return {"error": "No available models found"}
    
    # Run inference
    results = []
    total_questions = len(questions)
    for i, model in enumerate(available_models):
        logger.info(f"Running inference with {model.name} ({model.provider}) ({i+1}/{len(available_models)})")
        
        model_results = []
        correct_count = 0
        for j, question in enumerate(questions):
            record_key = (model.name, question["id"])
            if record_key in completed_records:
                logger.info(f"  Skipping question {question['id']} for {model.name} - already in JSONL results")
                continue
            if j % 10 == 0:
                logger.info(f"  Processing question {j+1}/{total_questions}")
            result = evaluator.evaluate_question(model.provider, model.model_id, question)
            
            # Only add to results and save if we got a valid response
            if result["response"] is not None:
                model_results.append(result)
                if result["is_correct"]:
                    correct_count += 1
                # Append to JSONL immediately
                if jsonl_path:
                    record = {
                        "model_name": model.name,
                        "provider": model.provider,
                        "model_id": model.model_id,
                        **result
                    }
                    with open(jsonl_path, 'a') as jf:
                        json.dump(record, jf)
                        jf.write('\n')
                    # Add to completed_records to prevent duplicates in the same run
                    completed_records.add(record_key)
            else:
                logger.warning(f"  Failed to get response for question {question['id']} - skipping")
        # Calculate metrics for summary (only for questions actually run)
        num_run = len(model_results)
        accuracy = correct_count / num_run if num_run > 0 else 0
        avg_response_time = sum(r["response_time"] for r in model_results if r["response_time"]) / num_run if num_run > 0 else 0
        model_summary = {
            "model_name": model.name,
            "provider": model.provider,
            "model_id": model.model_id,
            "total_questions": num_run,
            "correct_answers": correct_count,
            "accuracy": accuracy,
            "average_response_time": avg_response_time,
            "results": model_results
        }
        results.append(model_summary)
        logger.info(f"  {model.name}: {accuracy:.2%} accuracy, {avg_response_time:.2f}s avg response time")
    return {
        "timestamp": datetime.now().isoformat(),
        "data_path": data_path,
        "total_questions": total_questions,
        "models_tested": len(available_models),
        "results": results
    }

def generate_summary_from_jsonl(jsonl_path: str, output_path: str):
    """Generate a summary CSV report from JSONL data."""
    if not os.path.exists(jsonl_path):
        logger.warning(f"JSONL file {jsonl_path} not found, cannot generate summary")
        return
    
    # Read all records from JSONL and group by model
    model_stats = {}
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                model_name = record.get('model_name')
                if not model_name:
                    continue
                    
                if model_name not in model_stats:
                    model_stats[model_name] = {
                        'provider': record.get('provider', ''),
                        'total_questions': 0,
                        'correct_answers': 0,
                        'total_response_time': 0,
                        'valid_response_times': 0
                    }
                
                stats = model_stats[model_name]
                stats['total_questions'] += 1
                
                if record.get('is_correct', False):
                    stats['correct_answers'] += 1
                
                response_time = record.get('response_time')
                if response_time is not None:
                    stats['total_response_time'] += response_time
                    stats['valid_response_times'] += 1
                    
            except Exception as e:
                logger.warning(f"Error parsing JSONL line: {e}")
                continue
    
    # Generate summary data
    summary_data = []
    for model_name, stats in model_stats.items():
        accuracy = stats['correct_answers'] / stats['total_questions'] if stats['total_questions'] > 0 else 0
        avg_response_time = stats['total_response_time'] / stats['valid_response_times'] if stats['valid_response_times'] > 0 else 0
        
        summary_data.append({
            "Model": model_name,
            "Provider": stats['provider'],
            "Total_Questions": stats['total_questions'],
            "Correct_Answers": stats['correct_answers'],
            "Accuracy": f"{accuracy:.2%}",
            "Avg_Response_Time": f"{avg_response_time:.2f}s"
        })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path, index=False)
        logger.info(f"Summary report saved to {output_path}")
    else:
        logger.warning("No data found to generate summary report")

def main():
    parser = argparse.ArgumentParser(description="Run inference on IMCI MCQA data using AI Suite")
    parser.add_argument("--data", default="results/IMCI_qamc.json", help="Path to JSON data file")
    parser.add_argument("--models", nargs="+", default=[m.name for m in MODELS], 
                       help="List of model names to test")
    parser.add_argument("--max-questions", type=int, help="Maximum number of questions to test")
    parser.add_argument("--output-dir", default="results/inference", help="Output directory for results")
    parser.add_argument("--resume-incomplete", action="store_true", 
                       help="Only run models that have incomplete results")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter for incomplete models if requested
    models_to_run = args.models
    if args.resume_incomplete:
        # Load questions to get total count
        questions = load_questions(args.data)
        total_questions = len(questions)
        
        # Check which models are incomplete
        jsonl_path = output_dir / "inference_results.jsonl"
        completed_records = get_completed_records_from_jsonl(str(jsonl_path))
        
        incomplete_models = []
        for model_name in args.models:
            model_question_count = sum(1 for (m, q) in completed_records if m == model_name)
            if model_question_count < total_questions:
                incomplete_models.append(model_name)
                logger.info(f"Model {model_name} has {model_question_count}/{total_questions} questions - will resume")
        
        if not incomplete_models:
            logger.info("All specified models have complete results. Nothing to resume.")
            return
        
        models_to_run = incomplete_models
        logger.info(f"Resuming inference for {len(incomplete_models)} incomplete models: {incomplete_models}")
    
    # Run inference
    logger.info("Starting inference evaluation with AI Suite...")
    results = run_inference(args.data, models_to_run, args.max_questions, str(output_dir))
    
    if "error" in results:
        logger.error(f"Inference failed: {results['error']}")
        return
    
    # Generate summary from JSONL data
    jsonl_file = output_dir / "inference_results.jsonl"
    summary_file = output_dir / "inference_summary.csv"
    
    generate_summary_from_jsonl(str(jsonl_file), str(summary_file))
    
    # Print summary from CSV if it exists
    if summary_file.exists():
        print("\n" + "="*80)
        print("INFERENCE RESULTS SUMMARY")
        print("="*80)
        try:
            df = pd.read_csv(summary_file)
            for _, row in df.iterrows():
                print(f"{row['Model']:<20} | {row['Provider']:<10} | {row['Accuracy']:>6} | {row['Avg_Response_Time']:>6}")
        except Exception as e:
            logger.error(f"Error reading summary file: {e}")
        print("="*80)

if __name__ == "__main__":
    main() 