"""
Pratyora - one-time installer for offline Argos Translate language models.

Usage (needs internet once, then translation works offline):
    python install_translation_models.py            # default: Hindi + other Indian languages
    python install_translation_models.py hi kn ta   # only the languages you list

Each model translates <language> -> English.
"""

import sys

import argostranslate.package
import argostranslate.translate

DEFAULT_LANGS = ["hi", "bn", "ta", "te", "ml", "mr", "gu", "ur", "es", "fr", "de"]


def main(codes):
    print("Updating Argos package index...")
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()

    for code in codes:
        match = next((p for p in available if p.from_code == code and p.to_code == "en"), None)
        if match is None:
            print(f"[skip] No {code}->en package is available in the Argos index.")
            continue
        print(f"[install] {code}->en ...")
        argostranslate.package.install_from_path(match.download())

    installed = [l.code for l in argostranslate.translate.get_installed_languages()]
    print("Installed languages:", ", ".join(sorted(installed)))


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_LANGS)