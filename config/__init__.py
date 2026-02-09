import os
import json
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent


def load_config(config_path=None, domain=None):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file (optional).
        domain: Dataset name, overrides config (optional).
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if domain is not None:
        config['data']['domain'] = domain
    
    if 'data' in config:
        base_path = config['data'].get('base_path', './data')
        if isinstance(base_path, str) and base_path.startswith('./'):
            config['data']['base_path'] = str(PROJECT_ROOT / base_path[2:])
    
    if 'model' in config:
        for key, value in config['model'].items():
            if isinstance(value, str) and value.startswith('./'):
                config['model'][key] = str(PROJECT_ROOT / value[2:])
    
    config = _auto_adapt_hidden_dim(config)
    return config


def _auto_adapt_hidden_dim(config):
    """Set hidden_dim from dataset concept count if available."""
    try:
        domain = config['data']['domain']
        base_path = Path(config['data']['base_path'])
        stats_path = base_path / domain / 'statistic.json'
        
        if stats_path.exists():
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            
            num_concepts = stats.get('concepts_num')
            if num_concepts:
                if 'training' in config:
                    config['training']['hidden_dim'] = num_concepts
                if 'ecge' in config:
                    config['ecge']['hidden_dim'] = num_concepts
    except Exception:
        pass
    
    return config

def get_llm_config():
    """Get LLM configuration from environment variables"""
    return {
        'provider': os.getenv('LLM_PROVIDER', 'gpt'),
        'api_key': os.getenv('LLM_API_KEY', ''),
        'base_url': os.getenv('LLM_BASE_URL', 'https://api.openai.com/v1'),
        'model': os.getenv('LLM_MODEL', 'gpt-3.5-turbo')
    }


def get_kg_llm_config():
    """Get KG-specific LLM config from env (KG_LLM_*). Fallback: default LLM config."""
    kg_api_key = os.getenv('KG_LLM_API_KEY')
    
    if kg_api_key:
        return {
            'provider': os.getenv('KG_LLM_PROVIDER', 'gpt'),
            'api_key': kg_api_key,
            'base_url': os.getenv('KG_LLM_BASE_URL', 'https://api.openai.com/v1'),
            'model': os.getenv('KG_LLM_MODEL', 'gpt-4o')
        }
    return get_llm_config()
