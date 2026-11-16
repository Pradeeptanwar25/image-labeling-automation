from PIL import Image
from src.dataset_scanner import DatasetScanner

def test_scan_and_duplicate(tmp_path):
    d=tmp_path/'S25'; d.mkdir()
    im=Image.new('RGB',(100,50),'white'); im.save(d/'a.jpg')
    (d/'b.jpg').write_bytes((d/'a.jpg').read_bytes())
    report=DatasetScanner(tmp_path).scan()
    assert report['folder_counts']=={'S25':2}
    assert len(report['duplicates'])==1
    assert report['records'][0]['width']==100
