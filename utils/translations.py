import yaml
import os
from functools import lru_cache
from typing import Dict, Optional

class Translator:
    def __init__(self, config: dict = None):
        self.config = config
        self.default_locale = self.config['localization']['default_locale']
        self.current_locale = self.config['localization']['default_locale']
        self._load_translations()

    @lru_cache(maxsize=None)
    def _load_translations(self) -> Dict:
        """Load all translation files from translations directory"""
        translations = {}
        translation_dir = self.config['paths']['locale_dir']

        if not os.path.exists(translation_dir):
            os.makedirs(translation_dir)

        for filename in os.listdir(translation_dir):
            if filename.endswith('.yaml'):
                locale = filename.split('.')[0]
                with open(os.path.join(translation_dir, filename), 'r', encoding='utf-8') as f:
                    translations[locale] = yaml.safe_load(f)
        return translations

    def set_locale(self, locale: str) -> None:
        """Set the current locale"""
        if locale in self._load_translations():
            self.current_locale = locale
        else:
            self.current_locale = self.default_locale

    def get(self, key: str, **kwargs) -> str:
        """Get translated string for given key"""
        translations = self._load_translations()

        # Try to get translation for current locale
        try:
            text = translations[self.current_locale][key]
        except (KeyError, TypeError):
            # Fallback to default locale
            try:
                text = translations[self.default_locale][key]
            except (KeyError, TypeError):
                # Return key if no translation found
                return key

        # Apply string formatting if kwargs provided
        if kwargs:
            try:
                text = text.format(**kwargs)
            except KeyError:
                pass

        return text