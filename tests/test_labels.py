import json, pytest
from src.label_registry import LabelRegistry

def test_registry_and_mapping(tmp_path):
    labels=tmp_path/'labels.txt'; labels.write_text('a\nb\n',encoding='utf-8')
    mapping=tmp_path/'mapping.json'; mapping.write_text(json.dumps({'S25':'a'}),encoding='utf-8')
    r=LabelRegistry(labels,mapping).load()
    assert r.labels==['a','b']
    assert r.validate_mapping(['S25','S20'])['missing_folders']==['S20']

def test_duplicate_labels_fail(tmp_path):
    labels=tmp_path/'labels.txt'; labels.write_text('a\na\n',encoding='utf-8')
    with pytest.raises(ValueError): LabelRegistry(labels,tmp_path/'x.json').load()
