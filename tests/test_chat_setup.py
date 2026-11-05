import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "venv" / "01_chat.py"


def import_chat_script():
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv

    langchain_openai = types.ModuleType("langchain_openai")

    class DummyChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def invoke(self, question):
            return types.SimpleNamespace(content=f"stub reply for: {question}")

    langchain_openai.ChatOpenAI = DummyChatOpenAI
    sys.modules["langchain_openai"] = langchain_openai

    spec = importlib.util.spec_from_file_location("chat_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ChatSetupTests(unittest.TestCase):
    def test_repo_root_is_resolved_from_script(self):
        key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        try:
            module = import_chat_script()
            self.assertEqual(module.REPO_ROOT, ROOT)
            self.assertEqual(module.ENV_PATH, ROOT / ".env")
        finally:
            if key is None:
                os.environ.pop("OPENROUTER_API_KEY", None)
            else:
                os.environ["OPENROUTER_API_KEY"] = key

    def test_missing_api_key_raises_clear_error(self):
        key = os.environ.pop("OPENROUTER_API_KEY", None)
        try:
            module = import_chat_script()
            with self.assertRaises(RuntimeError):
                module.build_model()
        finally:
            if key is not None:
                os.environ["OPENROUTER_API_KEY"] = key


if __name__ == "__main__":
    unittest.main()
