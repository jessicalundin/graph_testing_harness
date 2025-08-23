import networkx as nx
import random
import json
import os
from pathlib import Path
from generate_graph import load_and_build_graph, create_node_type_map

# Paths - Mac-compatible using pathlib
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
QUESTIONS_PATH = PROJECT_ROOT / 'results' / 'IMCI_qamc.json'
GRAPH_PATH = PROJECT_ROOT / 'databases' / 'IMCI_schema.graphml'

def get_random_age_from_range(age_range):
    """Convert age range to a random specific age."""
    if age_range == '0-2':
        weeks = random.randint(0, 8)
        return f"{weeks} week"
    elif age_range == '2-60':
        months = random.randint(2, 60)
        if months < 23:
            return f"{months} month"
        return f"{int(months/12)} year"
    else:
        return age_range

def convert_age_to_months(age_string):
    """Convert age string to months for threshold calculations."""
    if "week" in age_string:
        weeks = int(age_string.split()[0])
        return weeks / 4.0  # Approximate weeks to months
    elif "month" in age_string:
        return int(age_string.split()[0])
    elif "year" in age_string:
        years = int(age_string.split()[0])
        return years * 12
    return 0

def format_age_specific_symptom(symptom, age_months):
    """Format symptom text based on specific age for clinical accuracy."""
    symptom_lower = symptom.lower()
    
    # Fast breathing thresholds based on IMCI guidelines
    if "fast breathing" in symptom_lower:
        if age_months < 2:
            # For 0-2 months range (infants)
            return "fast breathing (60 breaths per minute or more)"
        elif age_months <= 12:
            # For 2-12 months range
            return "fast breathing (50 breaths per minute or more)"
        else:
            # For 12 months to 5 years range
            return "fast breathing (40 breaths per minute or more)"
    
    # Add other age-specific symptoms as needed
    # Example for fever thresholds (if needed):
    # if "fever" in symptom_lower and "37.5" in symptom:
    #     return "fever (37.5°C or above)"
    
    # Return original symptom if no age-specific formatting needed
    return symptom

def get_distractors(correct, node_type, node_type_map, exclude=None, k=3):
    """Get k distractors of the same type, not including correct or exclude."""
    pool = set(node_type_map[node_type]) - {correct}
    if exclude:
        pool -= set(exclude)
    
    available = list(pool)
    if len(available) < k:
        print(f"Warning: Only {len(available)} distractor options available for {node_type}, need {k}")
        # Pad with generic options if needed to ensure we always have k distractors
        distractors = available.copy()
        
        # Add generic padding options to reach k distractors
        generic_options = []
        if node_type == 'Condition':
            generic_options = ["Other condition", "Alternative diagnosis", "Different condition"]
        elif node_type == 'Symptom':  
            generic_options = ["Other symptom", "Different symptom", "Alternative symptom"]
        elif node_type == 'Treatment':
            generic_options = ["Alternative treatment", "Different treatment", "Other treatment"]
        elif node_type == 'FollowUp':
            generic_options = ["Alternative follow-up", "Different follow-up", "Other follow-up"]
        
        # Add generic options until we have k distractors
        for generic in generic_options:
            if len(distractors) >= k:
                break
            if generic not in distractors and generic != correct:
                distractors.append(generic)
        
        return distractors[:k]  # Ensure we don't exceed k
    else:
        return random.sample(available, k)

def get_age_appropriate_distractors(correct, node_type, age_range, graph, node_type_map, k=3):
    """Get age-appropriate distractors based on node type and age range."""
    # Get all conditions with the same age range
    same_age_conditions = [c for c in node_type_map.get('Condition', []) 
                          if graph.nodes[c].get('age_range') == age_range]
    
    # Build pool based on node type
    age_appropriate_pool = set()
    
    if node_type == 'Condition':
        # For condition distractors, use conditions with same age range
        age_appropriate_pool = set(same_age_conditions) - {correct}
    
    elif node_type == 'Symptom':
        # For symptom distractors, get symptoms from same-age conditions
        for c in same_age_conditions:
            symptoms = [n for n in graph.predecessors(c) 
                       if graph.nodes[n]['type'] == 'Symptom']
            age_appropriate_pool.update(symptoms)
        age_appropriate_pool -= {correct}
    
    elif node_type == 'Treatment':
        # For treatment distractors, get treatments from same-age conditions
        for c in same_age_conditions:
            treatments = [n for n in graph.successors(c) 
                         if graph.nodes[n]['type'] == 'Treatment']
            age_appropriate_pool.update(treatments)
        age_appropriate_pool -= {correct}
    
    elif node_type == 'FollowUp':
        # For follow-up distractors, get follow-ups from same-age conditions
        for c in same_age_conditions:
            followups = [n for n in graph.successors(c) 
                        if graph.nodes[n]['type'] == 'FollowUp']
            age_appropriate_pool.update(followups)
        age_appropriate_pool -= {correct}
    
    # If we have enough age-appropriate distractors, use them
    if len(age_appropriate_pool) >= k:
        return random.sample(list(age_appropriate_pool), k)
    else:
        # Fall back to general pool if not enough age-appropriate options
        print(f"Warning: Only {len(age_appropriate_pool)} age-appropriate distractors for {node_type} (ages {age_range}), falling back to general pool")
        fallback_distractors = get_distractors(correct, node_type, node_type_map, k=k)
        
        # Ensure we always return exactly k distractors
        if len(fallback_distractors) < k:
            print(f"Error: Even fallback couldn't provide {k} distractors for {node_type}, got {len(fallback_distractors)}")
        
        return fallback_distractors

# Question templates for variety
# CONDITION_SYMPTOM_TEMPLATES: Given a condition, ask for symptoms (cond → symp)
CONDITION_SYMPTOM_TEMPLATES = [
    "Which symptom indicates {cond} in a {specific_age} old child?",
    "What is a key symptom of {cond} in a {specific_age} old child?",
    "A {specific_age} old child with {cond} would most likely present with which symptom?",
    "Which of the following symptoms suggests {cond} in a {specific_age} old child?"
]

# SYMPTOM_CONDITION_TEMPLATES: Given a symptom, ask for conditions (symp → cond)
SYMPTOM_CONDITION_TEMPLATES = [
    "A {specific_age} old child with {formatted_symptom} most likely has which condition?",
    "Which condition should you suspect in a {specific_age} old child presenting with {formatted_symptom}?",
    "A {specific_age} old child presenting with {formatted_symptom} is most likely to have:",
    "What is the most probable diagnosis for a {specific_age} old child with {formatted_symptom}?"
]

TREATMENT_TEMPLATES = [
    "How should you treat a {specific_age} old child with {cond}?",
    "What is the appropriate treatment for a {specific_age} old child diagnosed with {cond}?",
    "A {specific_age} old child with {cond} should receive which treatment?",
    "Which treatment is recommended for a {specific_age} old child with {cond}?"
]

FOLLOWUP_TEMPLATES = [
    "When should a {specific_age} old child with {cond} return for follow-up?",
    "What is the appropriate follow-up schedule for a {specific_age} old child with {cond}?",
    "A {specific_age} old child treated for {cond} should return for follow-up:",
    "When should you schedule the next visit for a {specific_age} old child with {cond}?"
]

SEVERITY_TEMPLATES = [
    "How severe is {cond} in a {specific_age} old child?",
    "What is the severity classification of {cond} in a {specific_age} old child?",
    "A {specific_age} old child with {cond} should be classified as:",
    "The severity level of {cond} in a {specific_age} old child is typically:"
]

def create_question(qid, question_text, answer, label, options, template_id=None):
    """Create a standardized question dictionary."""
    # Ensure options are unique while preserving order
    seen = set()
    unique_options = []
    for opt in options:
        if opt not in seen:
            seen.add(opt)
            unique_options.append(opt)
    
    # If we have duplicates, log a warning
    if len(unique_options) < len(options):
        print(f"Warning: Duplicate options found in question {qid}, removed {len(options) - len(unique_options)} duplicates")
    
    # Ensure we have at least 2 options (answer + at least one distractor)
    if len(unique_options) < 2:
        print(f"Error: Question {qid} has insufficient unique options: {unique_options}")
        return None
    
    letter_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    correct_letter = letter_map[unique_options.index(answer)]
    
    question_dict = {
        'id': qid,
        'question': question_text,
        'answer': answer,
        'label': label,
        'options': {letter_map[i]: opt for i, opt in enumerate(unique_options)},
        'correct_answer': correct_letter
    }
    
    # Add template ID if provided
    if template_id is not None:
        question_dict['template_id'] = template_id
    
    return question_dict

def generate_symptom_questions(cond, symptoms, age_range, node_type_map, graph):
    """Generate symptom-related questions for a condition."""
    questions = []
    
    for symp in symptoms:
        # Generate a new random age for each question
        specific_age = get_random_age_from_range(age_range)
        
        # Question 1: What is a symptom of condition X? (cond → symp)
        # Use age-appropriate symptom distractors
        distractors = get_age_appropriate_distractors(symp, 'Symptom', age_range, graph, node_type_map)
        
        # Format age-specific symptom for clinical accuracy
        age_months = convert_age_to_months(specific_age)
        formatted_symptom = format_age_specific_symptom(symp, age_months)
        
        # Format distractors too for consistency
        formatted_distractors = [format_age_specific_symptom(d, age_months) for d in distractors]
        options = formatted_distractors + [formatted_symptom]
        random.shuffle(options)
        
        if len(options) < 4:
            print(f"Warning: Symptom question for {cond} only has {len(options)} options instead of 4")
        
        # Randomly select template
        template_idx = random.randint(0, len(CONDITION_SYMPTOM_TEMPLATES) - 1)
        question_text = CONDITION_SYMPTOM_TEMPLATES[template_idx].format(
            cond=cond, specific_age=specific_age
        )
        
        question1 = create_question(
            qid=len(questions) + 1,
            question_text=question_text,
            answer=formatted_symptom,
            label='cond_symp',
            options=options,
            template_id=f'cond_symp_t{template_idx + 1}'
        )
        if question1:
            questions.append(question1)
        
        # Generate another random age for the second question
        specific_age2 = get_random_age_from_range(age_range)
        
        # Question 2: A child has symptom X. What is the condition? (symp → cond)
        # Use age-appropriate condition distractors
        cond_distractors = get_age_appropriate_distractors(cond, 'Condition', age_range, graph, node_type_map)
        cond_options = cond_distractors + [cond]
        random.shuffle(cond_options)
        
        # Format age-specific symptom for the question text
        age_months2 = convert_age_to_months(specific_age2)
        formatted_symptom2 = format_age_specific_symptom(symp, age_months2)
        
        if len(cond_options) < 4:
            print(f"Warning: Condition question for symptom {symp} only has {len(cond_options)} options instead of 4")
        
        # Randomly select template
        template_idx2 = random.randint(0, len(SYMPTOM_CONDITION_TEMPLATES) - 1)
        question_text2 = SYMPTOM_CONDITION_TEMPLATES[template_idx2].format(
            specific_age=specific_age2, formatted_symptom=formatted_symptom2
        )
        
        question2 = create_question(
            qid=len(questions) + 1,
            question_text=question_text2,
            answer=cond,
            label='symp_cond',
            options=cond_options,
            template_id=f'symp_cond_t{template_idx2 + 1}'
        )
        if question2:
            questions.append(question2)
        
    
    return questions

def generate_treatment_questions(cond, treatments, node_type_map, age_range, graph):
    """Generate treatment questions for a condition."""
    questions = []
    
    for treat in treatments:
        # Generate a new random age for each question
        specific_age = get_random_age_from_range(age_range)
        
        # Use age-appropriate treatment distractors
        distractors = get_age_appropriate_distractors(treat, 'Treatment', age_range, graph, node_type_map)
        options = distractors + [treat]
        random.shuffle(options)
        
        if len(options) < 4:
            print(f"Warning: Treatment question for {cond} only has {len(options)} options instead of 4")
        
        # Randomly select template
        template_idx = random.randint(0, len(TREATMENT_TEMPLATES) - 1)
        question_text = TREATMENT_TEMPLATES[template_idx].format(
            specific_age=specific_age, cond=cond
        )
        
        question = create_question(
            qid=len(questions) + 1,
            question_text=question_text,
            answer=treat,
            label='cond_treat',
            options=options,
            template_id=f'cond_treat_t{template_idx + 1}'
        )
        if question:
            questions.append(question)
        
    
    return questions

def generate_followup_questions(cond, followups, node_type_map, age_range, graph):
    """Generate follow-up questions for a condition."""
    questions = []
    
    for follow in followups:
        # Generate a new random age for each question
        specific_age = get_random_age_from_range(age_range)
        
        # Use age-appropriate follow-up distractors
        distractors = get_age_appropriate_distractors(follow, 'FollowUp', age_range, graph, node_type_map)
        options = distractors + [follow]
        random.shuffle(options)
        
        if len(options) < 4:
            print(f"Warning: Follow-up question for {cond} only has {len(options)} options instead of 4")
        
        # Randomly select template
        template_idx = random.randint(0, len(FOLLOWUP_TEMPLATES) - 1)
        question_text = FOLLOWUP_TEMPLATES[template_idx].format(
            specific_age=specific_age, cond=cond
        )
        
        question = create_question(
            qid=len(questions) + 1,
            question_text=question_text,
            answer=follow,
            label='cond_followup',
            options=options,
            template_id=f'cond_followup_t{template_idx + 1}'
        )
        if question:
            questions.append(question)
        
    
    return questions

def generate_severity_questions(cond, severities, node_type_map, age_range):
    """Generate severity questions for a condition."""
    questions = []
    
    for sev in severities:
        # Generate a new random age for each question
        specific_age = get_random_age_from_range(age_range)
        
        # Get severity distractors from the graph (no age filtering for severity)
        distractors = get_distractors(sev, 'Severity', node_type_map, k=2)
        
        # Add "none" as an additional distractor option for severity questions
        distractors.append('none')
        options = distractors + [sev]
        random.shuffle(options)
        
        if len(options) < 4:
            print(f"Warning: Severity question for {cond} only has {len(options)} options instead of 4")
        
        # Randomly select template
        template_idx = random.randint(0, len(SEVERITY_TEMPLATES) - 1)
        question_text = SEVERITY_TEMPLATES[template_idx].format(
            cond=cond, specific_age=specific_age
        )
        
        question = create_question(
            qid=len(questions) + 1,
            question_text=question_text,
            answer=sev,
            label='cond_severity',
            options=options,
            template_id=f'cond_severity_t{template_idx + 1}'
        )
        if question:
            questions.append(question)
        
    
    return questions

def generate_all_questions(graph, node_type_map):
    """Generate all questions for all conditions."""
    all_questions = []
    qid = 1
    
    for cond in node_type_map.get('Condition', []):
        age_range = graph.nodes[cond].get('age_range', '2-60')
        
        # Get all related nodes for this condition
        symptoms = [n for n in graph.predecessors(cond) if graph.nodes[n]['type'] == 'Symptom']
        treatments = [n for n in graph.successors(cond) if graph.nodes[n]['type'] == 'Treatment']
        followups = [n for n in graph.successors(cond) if graph.nodes[n]['type'] == 'FollowUp']
        severities = [n for n in graph.successors(cond) if graph.nodes[n]['type'] == 'Severity']
        
        # Generate questions for each type
        symp_questions = generate_symptom_questions(
            cond, symptoms, age_range, node_type_map, graph)
        treat_questions = generate_treatment_questions(cond, treatments, node_type_map, age_range, graph)
        follow_questions = generate_followup_questions(cond, followups, node_type_map, age_range, graph)
        sev_questions = generate_severity_questions(cond, severities, node_type_map, age_range)
        
        # Combine all questions
        all_questions.extend(symp_questions)
        all_questions.extend(treat_questions)
        all_questions.extend(follow_questions)
        all_questions.extend(sev_questions)
    
    # Reassign IDs sequentially
    for i, question in enumerate(all_questions, 1):
        question['id'] = i
    
    return all_questions

def save_questions(questions):
    """Save questions to JSON file."""
    # Save questions as JSON
    os.makedirs(os.path.dirname(QUESTIONS_PATH), exist_ok=True)
    with open(QUESTIONS_PATH, 'w') as f:
        json.dump(questions, f, indent=4)
    print(f"Questions saved to {QUESTIONS_PATH}")

def load_existing_graph():
    """Load existing graph from GraphML file if it exists, otherwise build new one."""
    if os.path.exists(GRAPH_PATH):
        print(f"Loading existing graph from {GRAPH_PATH}")
        return nx.read_graphml(GRAPH_PATH)
    else:
        print("No existing graph found, building new one...")
        return load_and_build_graph()

def main():
    """Main function to generate questions."""
    print("Loading graph...")
    graph = load_existing_graph()
    
    print("Creating node type mapping...")
    node_type_map = create_node_type_map(graph)
    
    print("Generating questions...")
    questions = generate_all_questions(graph, node_type_map)
    
    print("Saving questions...")
    save_questions(questions)
    
    print(f"Generated {len(questions)} questions.")

if __name__ == "__main__":
    main()