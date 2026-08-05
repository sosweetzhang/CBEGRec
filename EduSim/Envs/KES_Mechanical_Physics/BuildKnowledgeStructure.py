
import os
from pathlib import Path
from openai import OpenAI
import json



class BuildKG_LLM:
    def __init__(self, model_name):
        self.model_name = model_name

    
    def KG_generation(self, concepts):

        
        prompt = f"""
            You are a knowledgeable teacher tasked with constructing a directed acyclic knowledge graph of prerequisite relationships between the following concepts. 
            Based on the list of concepts provided below, identify the prerequisite concepts for each item and return the result in JSON format.

            The structure of the JSON should be as follows:
            
              "ConceptID1": ["PrerequisiteConceptID1", "PrerequisiteConceptID2", ...],
              "ConceptID2": ["PrerequisiteConceptID1", ...],
              ...
            
            List of Concepts (including names and their corresponding IDs): {concepts}
            
            Please ensure that:
            1. The knowledge graph is a Directed Acyclic Graph (DAG), meaning that no concept should have circular dependencies (no cycles are allowed).
            2. Each concept can have zero or more prerequisites, but no concept should have more than 5 prerequisite concepts. If a concept has more than 5, only include the 5 most relevant prerequisites based on logical dependency.
            3. Concept IDs must be represented as strings, e.g., "1" for "Rod Problems".
            4. If a concept has no prerequisites, its prerequisite list should be an empty array [].
            5. The relationships should strictly follow the logical order of concept understanding, ensuring no backward dependencies.
            
            Return a strictly valid and complete JSON object with no additional explanations or extra output.
            """

        api_key = os.getenv("KG_LLM_API_KEY") or os.getenv("LLM_API_KEY", "")
        base_url = os.getenv("KG_LLM_BASE_URL") or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        client = OpenAI(api_key=api_key, base_url=base_url)

        messages = [
            {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
            {"role": "user", "content": prompt},
        ]

        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=messages
        )

        outputs = completion.choices[0].message.content.split("\n")

        json_string = ''.join(outputs)

        json_data = json.loads(json_string)

        return json_data


if __name__ == '__main__':
    model_name = os.getenv("KG_LLM_MODEL", "qwen-plus")
    KG_LLM = BuildKG_LLM(model_name)
    base_dir = Path(__file__).resolve().parent
    concepts_file_path = base_dir / "meta_data" / "concept2id.json"
    if not concepts_file_path.exists():
        concepts_file_path = Path("data/Mechanical_Physics/concept2id.json")
    concepts = []
    with open(concepts_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and "," in line:
                concepts.append(line)
    KG_structure = KG_LLM.KG_generation(concepts)
    save_file = base_dir / "meta_data" / "KG_structure.json"
    save_file.parent.mkdir(parents=True, exist_ok=True)
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(KG_structure, f, ensure_ascii=False, indent=2)
    print(KG_structure)

