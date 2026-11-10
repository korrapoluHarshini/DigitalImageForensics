"""Independent analysis modules.

Each module exposes a small, UI-independent API and is imported explicitly so
that a missing optional dependency in one module cannot prevent the others from
loading:

* ``modules.image_manipulation``     - MVSS-Net forgery localization
* ``modules.ai_generated_detection`` - pluggable AI-image detector
* ``modules.chat_tampering``         - screenshot screening + OCR

Structured results are returned by ``detect_image_manipulation``,
``detect_ai_generated_image`` and ``detect_chat_tampering``.
"""
