# Chat screenshot detector

Module 3 reuses the MVSS-Net forgery-localization checkpoint for visual
screening and adds an independent OCR + region/text-association pipeline. No
separate chat model is required for the application to work.

If a dedicated chat-screenshot model becomes available later, integrate it in
`modules/chat_tampering/inference.py` behind the same
`ChatTamperingResult` interface, or supply it as a replacement localization
backend. Keep the OCR and reporting layers unchanged so the two visual
pipelines can still be evaluated independently.
