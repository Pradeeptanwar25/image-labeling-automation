from pathlib import Path
import json

class LabelRegistry:
    def __init__(self, label_file: str | Path, mapping_file: str | Path):
        self.label_file = Path(label_file)
        self.mapping_file = Path(mapping_file)
        self.labels: list[str] = []
        self.mapping: dict[str, str] = {}

    def load(self):
        if not self.label_file.is_file():
            raise FileNotFoundError(f"Master label file not found: {self.label_file}")
        raw = self.label_file.read_text(encoding="utf-8-sig")
        labels=[]
        for line in raw.splitlines():
            value=line.strip().lstrip("-*").strip()
            if value and not value.startswith("#"):
                labels.append(value)
        duplicates=sorted({x for x in labels if labels.count(x)>1})
        if duplicates:
            raise ValueError(f"Duplicate labels: {duplicates}")
        if not labels:
            raise ValueError("Master label file is empty")
        self.labels=labels
        if self.mapping_file.exists():
            data=json.loads(self.mapping_file.read_text(encoding="utf-8"))
            if not isinstance(data,dict):
                raise ValueError("Folder mapping must be a JSON object")
            self.mapping={str(k):str(v) for k,v in data.items()}
        return self

    def validate_mapping(self, folders):
        folders=sorted(set(folders))
        missing=[f for f in folders if f not in self.mapping]
        invalid={f:self.mapping[f] for f in folders if f in self.mapping and self.mapping[f] not in self.labels}
        return {'missing_folders':missing,'invalid_labels':invalid}

    def save_mapping(self, mapping):
        invalid={k:v for k,v in mapping.items() if v not in self.labels}
        if invalid:
            raise ValueError(f"Mappings use labels absent from master list: {invalid}")
        self.mapping_file.parent.mkdir(parents=True,exist_ok=True)
        self.mapping_file.write_text(json.dumps(mapping,indent=2,ensure_ascii=False),encoding='utf-8')
        self.mapping=dict(mapping)
