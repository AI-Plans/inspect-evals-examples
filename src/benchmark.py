from inspect_ai import Task, task, eval
from inspect_ai.dataset import Dataset, Sample
from inspect_ai.solver import generate
from inspect_ai.scorer import model_graded_fact
from datasets import load_dataset
from config import GROQ_API_KEY, MODEL_NAME, PROVIDER, DATASET

class HHHDataset(Dataset):
    def __init__(self):
        self.examples = []
        dataset = load_dataset(DATASET, trust_remote_code=True)
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
    def location(self): return DATASET
    @property
    def shuffled(self): return False

@task
def hhh_benchmark():
    dataset = HHHDataset()
    return Task(
        dataset=dataset,
        solver=generate(
            model=MODEL_NAME,
            provider=PROVIDER,
            api_key=GROQ_API_KEY
        ),
        scorer=model_graded_fact()
    )

if __name__ == "__main__":
    task = hhh_benchmark()
    results = eval(task, model= MODEL_NAME)
    print(results)
