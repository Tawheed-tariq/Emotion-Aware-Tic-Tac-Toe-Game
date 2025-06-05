import yaml

def load_config(path='configs/config.yaml'):
    with open(path, 'r') as file:
        return yaml.safe_load(file)
