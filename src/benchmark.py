""""
from inspect_ai import Task, task
from inspect_ai.dataset import Dataset
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate
from datasets import load_dataset
from config import GROQ_API_KEY, MODEL_NAME

def load_hhh_dataset():
    dataset = load_dataset("HuggingFaceH4/hhh_alignment")
    return Dataset(
        inputs=[ex["input"] for ex in dataset["train"]], 
        targets=[ex["target"] for ex in dataset["train"]]
    )

@task
def hhh_benchmark():
    return Task(
        dataset=load_hhh_dataset(),
        solver=generate(model=MODEL_NAME, api_key=GROQ_API_KEY),
        scorer=model_graded_fact(model=MODEL_NAME)
    )

if __name__ == "__main__":
    hhh_benchmark()
"""


"""
from inspect_ai import Task, task
from inspect_ai.dataset import Dataset
from inspect_ai.solver import generate
from inspect_ai.scorer import model_graded_fact
from datasets import load_dataset
from config import GROQ_API_KEY, MODEL_NAME

def load_hhh_dataset():
    # Load with trust remote code since HHH requires it
    dataset = load_dataset("HuggingFaceH4/hhh_alignment", trust_remote_code=True)
    
    # Each example needs a 'prompt' and 'completion' field for inspect-ai
    formatted_data = []
    
    # HHH has categories instead of train/test splits
    for category in ['harmless', 'helpful', 'honest', 'other']:
        category_data = dataset[category]
        formatted_data.extend([{
            'prompt': item['prompt'],  # The input text
            'completion': item['target']  # The expected output
        } for item in category_data])
    
    # Create Dataset with properly formatted examples
    return Dataset.from_examples(formatted_data)

@task
def hhh_benchmark():
    dataset = load_hhh_dataset()
    return Task(
        dataset=dataset,
        solver=generate(
            model="llama2-70b", 
            provider="groq",
            api_key=GROQ_API_KEY
        ),
        scorer=model_graded_fact()
    )

if __name__ == "__main__":
    hhh_benchmark()

"""

"""
from inspect_ai import Task, task
from inspect_ai.dataset import Dataset
from inspect_ai.solver import generate
from inspect_ai.scorer import model_graded_fact
from datasets import load_dataset
from config import GROQ_API_KEY, MODEL_NAME

def load_hhh_dataset():
    # Load with trust remote code and print the structure
    dataset = load_dataset("HuggingFaceH4/hhh_alignment", trust_remote_code=True)
    print("Dataset keys:", dataset.keys())  # Let's see what we actually have
    
    # Format for the first available split
    first_split = list(dataset.keys())[0]  # Get the first available split
    raw_data = dataset[first_split]
    
    formatted_data = [{
        'prompt': item['prompt'],
        'completion': item['target'] 
    } for item in raw_data]
    
    return Dataset.from_examples(formatted_data)

@task
def hhh_benchmark():
    dataset = load_hhh_dataset()
    return Task(
        dataset=dataset,
        solver=generate(
            model="llama2-70b", 
            provider="groq",
            api_key=GROQ_API_KEY
        ),
        scorer=model_graded_fact()
    )

if __name__ == "__main__":
    hhh_benchmark()

"""

"""
from inspect_ai import Task, task
from inspect_ai.dataset import Dataset
from inspect_ai.solver import generate
from inspect_ai.scorer import model_graded_fact
from datasets import load_dataset
from config import GROQ_API_KEY, MODEL_NAME

def load_hhh_dataset():
    # Load dataset
    dataset = load_dataset("HuggingFaceH4/hhh_alignment", trust_remote_code=True)
    test_data = dataset['test']
    
    # Let's see the actual structure of one example
    print("Example data structure:", test_data[0])
    
    formatted_data = [{
        'prompt': item['input'],  # Changed from 'prompt' to 'input'
        'completion': item['target']
    } for item in test_data]
    
    return Dataset.from_examples(formatted_data)

@task
def hhh_benchmark():
    dataset = load_hhh_dataset()
    return Task(
        dataset=dataset,
        solver=generate(
            model="llama2-70b", 
            provider="groq",
            api_key=GROQ_API_KEY
        ),
        scorer=model_graded_fact()
    )

if __name__ == "__main__":
    hhh_benchmark()
"""

"""
from inspect_ai import Task, task
from inspect_ai.dataset import Dataset
from inspect_ai.solver import generate
from inspect_ai.scorer import model_graded_fact
from datasets import load_dataset
from config import GROQ_API_KEY, MODEL_NAME

def load_hhh_dataset():
    dataset = load_dataset("HuggingFaceH4/hhh_alignment", trust_remote_code=True)
    test_data = dataset['test']
    
    formatted_data = [{
        'prompt': item['input'],
        'completion': item['targets']['choices'][0]  # Taking the first choice which has label 1
    } for item in test_data]
    
    return Dataset.from_examples(formatted_data)

@task
def hhh_benchmark():
    dataset = load_hhh_dataset()
    return Task(
        dataset=dataset,
        solver=generate(
            model="llama2-70b", 
            provider="groq",
            api_key=GROQ_API_KEY
        ),
        scorer=model_graded_fact()
    )

if __name__ == "__main__":
    hhh_benchmark()
"""

"""
from inspect_ai import Task, task
from inspect_ai.dataset import Dataset
from inspect_ai.solver import generate
from inspect_ai.scorer import model_graded_fact
from datasets import load_dataset
from config import GROQ_API_KEY, MODEL_NAME

class HHHDataset(Dataset):
    def __init__(self):
        self.examples = []
        self._load_data()
        
    def _load_data(self):
        dataset = load_dataset("HuggingFaceH4/hhh_alignment", trust_remote_code=True)
        test_data = dataset['test']
        
        for item in test_data:
            self.examples.append({
                'input': item['input'],
                'target': item['targets']['choices'][0]  # Taking the first choice (label 1)
            })

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)

    def filter(self, predicate):
        filtered = [ex for ex in self.examples if predicate(ex)]
        new_dataset = HHHDataset()
        new_dataset.examples = filtered
        return new_dataset

    def shuffle(self):
        import random
        random.shuffle(self.examples)

    def shuffled(self):
        import random
        new_dataset = HHHDataset()
        new_dataset.examples = self.examples.copy()
        new_dataset.shuffle()
        return new_dataset

    def sort(self, key):
        self.examples.sort(key=key)

    @property
    def name(self):
        return "HHH Alignment Dataset"

    @property
    def location(self):
        return "HuggingFaceH4/hhh_alignment"

@task
def hhh_benchmark():
    dataset = HHHDataset()
    return Task(
        dataset=dataset,
        solver=generate(
            model="llama2-70b", 
            provider="groq",
            api_key=GROQ_API_KEY
        ),
        scorer=model_graded_fact()
    )

if __name__ == "__main__":
    hhh_benchmark()
"""


"""
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Dataset
from inspect_ai.solver import generate
from inspect_ai.scorer import model_graded_fact
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Example:
    input: str
    choices: List[str]
    labels: List[int]
    id: str
    sandbox: None = None
    metadata: Dict = field(default_factory=dict)

class HHHDataset(Dataset):
    def __init__(self):
        self.examples = []
        dataset = load_dataset("HuggingFaceH4/hhh_alignment", trust_remote_code=True)
        for i, item in enumerate(dataset['test']):
            self.examples.append(Example(
                input=item['input'],
                choices=item['targets']['choices'],
                labels=item['targets']['labels'],
                id=f"example_{i}"
            ))

    def __getitem__(self, idx): return self.examples[idx]
    def __len__(self): return len(self.examples)
    def filter(self, predicate): return [ex for ex in self.examples if predicate(ex)]
    def shuffle(self): pass
    def sort(self, key): pass
    @property
    def name(self): return "HHH"
    @property
    def location(self): return "HuggingFaceH4/hhh_alignment"
    @property
    def shuffled(self): return False

@task
def hhh_benchmark():
    dataset = HHHDataset()
    return Task(
        dataset=dataset,
        solver=generate(model="groq/llama2-70b"),
        scorer=model_graded_fact()
    )

if __name__ == "__main__":
    task = hhh_benchmark()
    results = eval(task, model="groq/llama2-70b")
    print(results)
"""


"""
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Dataset
from inspect_ai.solver import generate
from inspect_ai.scorer import model_graded_fact
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Example:
    input: str
    target: str  # Added target field
    choices: List[str]
    labels: List[int]
    id: str
    sandbox: None = None
    metadata: Dict = field(default_factory=dict)
    model_copy: bool = False  # Added model_copy attribute
    files: List = field(default_factory=list)  # Added files attribute

class HHHDataset(Dataset):
    def __init__(self):
        self.examples = []
        dataset = load_dataset("HuggingFaceH4/hhh_alignment", trust_remote_code=True)
        for i, item in enumerate(dataset['test']):
            # Use the choice with label 1 (good response) as target
            target_idx = item['targets']['labels'].index(1)
            target = item['targets']['choices'][target_idx]
            
            self.examples.append(Example(
                input=item['input'],
                target=target,  # Added target
                choices=item['targets']['choices'],
                labels=item['targets']['labels'],
                id=f"example_{i}"
            ))

    def __getitem__(self, idx): return self.examples[idx]
    def __len__(self): return len(self.examples)
    def filter(self, predicate): return [ex for ex in self.examples if predicate(ex)]
    def shuffle(self): pass
    def sort(self, key): pass
    @property
    def name(self): return "HHH"
    @property
    def location(self): return "HuggingFaceH4/hhh_alignment"
    @property
    def shuffled(self): return False

@task
def hhh_benchmark():
    dataset = HHHDataset()
    return Task(
        dataset=dataset,
        solver=generate(model="groq/llama2-70b"),
        scorer=model_graded_fact()
    )

if __name__ == "__main__":
    task = hhh_benchmark()
    results = eval(task, model="groq/llama2-70b")
    print(results)
"""


from inspect_ai import Task, task, eval
from inspect_ai.dataset import Dataset, Sample
from inspect_ai.solver import generate
from inspect_ai.scorer import model_graded_fact
from datasets import load_dataset
from config import GROQ_API_KEY  # Added this import

class HHHDataset(Dataset):
    def __init__(self):
        self.examples = []
        dataset = load_dataset("HuggingFaceH4/hhh_alignment", trust_remote_code=True)
        for i, item in enumerate(dataset['test']):
            target_idx = item['targets']['labels'].index(1)
            self.examples.append(Sample(
                id=f"example_{i}",
                input=item['input'],
                target=item['targets']['choices'][target_idx]
            ))

    def __getitem__(self, idx): return self.examples[idx]
    def __len__(self): return len(self.examples)
    def filter(self, predicate): return [ex for ex in self.examples if predicate(ex)]
    def shuffle(self): pass
    def sort(self, key): pass
    @property
    def name(self): return "HHH"
    @property
    def location(self): return "HuggingFaceH4/hhh_alignment"
    @property
    def shuffled(self): return False

@task
def hhh_benchmark():
    dataset = HHHDataset()
    return Task(
        dataset=dataset,
        solver=generate(
            #model="groq/llama2-70b-4096",
            model="groq/llama-3.1-8b-instant",
            provider="groq",
            api_key=GROQ_API_KEY  # Added API key here
        ),
        scorer=model_graded_fact()
    )

if __name__ == "__main__":
    task = hhh_benchmark()

    results = eval(task, model="groq/llama-3.1-8b-instant")  # Added model back
    #results = eval(task)  # Remove model parameter from here as it's already in the solver
    print(results)
