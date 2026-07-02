"""Chọn model cho lab — Groq (qua LiteLLM) nếu có GROQ_API_KEY, ngược lại Gemini.

Đặt trong .env:
    GROQ_API_KEY=gsk_...
    GROQ_MODEL=llama-3.3-70b-versatile   # tùy chọn, mặc định llama-3.3-70b-versatile
Xóa/bỏ trống GROQ_API_KEY để quay về Gemini.
"""

from __future__ import annotations

import os

from lab_utils.env_setup import load_lab_env

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


def get_lab_model():
    """Trả về model cho LlmAgent: LiteLlm(groq/...) hoặc chuỗi tên model Gemini."""
    load_lab_env()
    if os.getenv("GROQ_API_KEY"):
        from google.adk.models.lite_llm import LiteLlm

        model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        # temperature=0: Llama trên Groq hay sinh tool-call sai định dạng ở
        # temperature cao ("Failed to call a function")
        return LiteLlm(model=f"groq/{model_name}", temperature=0)
    return DEFAULT_GEMINI_MODEL
