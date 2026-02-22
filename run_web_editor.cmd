@echo off
setlocal
uvicorn web_editor.server:app --host 127.0.0.1 --port 8000 --reload
