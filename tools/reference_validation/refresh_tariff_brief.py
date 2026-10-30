"""Regenerate the bilingual brief, figures and frozen outputs from actual solves."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from puremacro.examples.first_brief_tariffs import run_brief, write_outputs, page_block


def refresh(output):
    study = run_brief(output / 'en', experiment='sigma=2')
    write_outputs(study['solution'], study['households'], output / 'es',
                  lang='es', experiment=study['experiment'])
    fixture = Path('tests/fixtures/first_brief_tariffs')
    for name in ('headline.json', 'manifest.json'):
        shutil.copyfile(output / 'en' / name, fixture / name)
    for lang in ('en', 'es'):
        page = Path('docs') / ('es' if lang == 'es' else '') / 'first_brief_tariffs.md'
        text = page.read_text(encoding='utf-8')
        for block in ('headline', 'headline_reading', 'households', 'experiment', 'experiment_reading'):
            pattern = rf'(<!-- first-brief:{block} -->\n).*?(\n<!-- /first-brief:{block} -->)'
            body = page_block(study['record'], block, lang)
            text, count = re.subn(pattern, lambda match: match[1]+body+match[2], text, flags=re.S)
            if count != 1:
                raise ValueError(f'missing or duplicate {block} block in {page}')
        page.write_text(text, encoding='utf-8')
        shutil.copyfile(output / lang / 'first_brief.png', Path('docs/assets/first_brief_tariffs') / f'first_brief_{lang}.png')
    print(json.dumps(study['record']['national'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    refresh(parser.parse_args().output)
