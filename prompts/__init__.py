from pathlib import Path
from typing import Dict, List, Tuple

import yaml


PROMPT_ROOT = Path(__file__).resolve().parent
DEFAULT_LANG = "en"


class _SafeDict(dict):
    def __missing__(self, key):
        return ""


def get_prompt_lang(config: dict = None) -> str:
    if not config:
        return DEFAULT_LANG
    prompt_config = config.get("prompt", {})
    return prompt_config.get("language", DEFAULT_LANG)


def _load_templates(lang: str) -> Dict:
    prompt_path = PROMPT_ROOT / f"{lang}.yaml"
    if not prompt_path.exists():
        prompt_path = PROMPT_ROOT / f"{DEFAULT_LANG}.yaml"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _format(template: str, values: Dict) -> str:
    return template.format_map(_SafeDict(values))


def get_prompt(prompt_name: str, lang: str = DEFAULT_LANG, **kwargs) -> Tuple[str, str]:
    templates = _load_templates(lang)
    style_aliases = {
        "teacher_style_ref": ("teacher", "style_ref"),
        "teacher_style_free": ("teacher", "style_free"),
    }
    if prompt_name in style_aliases:
        section_name, key = style_aliases[prompt_name]
        return "", _format(templates[section_name][key], kwargs)
    if prompt_name not in templates:
        raise KeyError(f"Unknown prompt template: {prompt_name}")
    section = templates[prompt_name]
    system = _format(section.get("system", ""), kwargs)
    user = _format(section.get("user", ""), kwargs)
    return system, user


def build_messages(prompt_name: str, config: dict = None, **kwargs) -> List[Dict[str, str]]:
    aliases = {
        "bundle_selection_reason": "j_t",
        "exercise": "e_t",
        "ref_answer": "a_ref_t",
        "sol_answer": "a_sol_t",
        "exercise_explanation": "J_t",
        "mastery_levels": "h_t",
    }
    for old_name, paper_name in aliases.items():
        if old_name in kwargs and paper_name not in kwargs:
            kwargs[paper_name] = kwargs[old_name]
    lang = get_prompt_lang(config)
    system, user = get_prompt(prompt_name, lang=lang, **kwargs)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages
