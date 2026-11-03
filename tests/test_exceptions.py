"""handle_exceptions 装饰器的回归测试"""

import unittest
from unittest.mock import patch

from src.core.exceptions import TradingSystemException, handle_exceptions


class TestHandleExceptions(unittest.TestCase):
    def test_returns_default_and_logs(self):
        # 修复前：捕获到异常时装饰器自己先抛 UnboundLocalError
        @handle_exceptions(default_return="fallback")
        def boom():
            raise ValueError("bad")

        with patch("src.core.logging.log_error") as log_error:
            self.assertEqual(boom(), "fallback")
        log_error.assert_called_once()
        message, exc = log_error.call_args.args
        self.assertIn("boom", message)
        self.assertIsInstance(exc, ValueError)

    def test_no_logging_when_disabled(self):
        @handle_exceptions(default_return=0, log_error=False)
        def boom():
            raise ValueError("bad")

        with patch("src.core.logging.log_error") as log_error:
            self.assertEqual(boom(), 0)
        log_error.assert_not_called()

    def test_reraise(self):
        @handle_exceptions(reraise=True, log_error=False)
        def boom():
            raise TradingSystemException("bad")

        with self.assertRaises(TradingSystemException):
            boom()

    def test_unlisted_exception_propagates(self):
        @handle_exceptions(exception_types=(KeyError,), log_error=False)
        def boom():
            raise ValueError("bad")

        with self.assertRaises(ValueError):
            boom()

    def test_success_path_and_metadata(self):
        @handle_exceptions()
        def ok(x):
            """docstring"""
            return x * 2

        self.assertEqual(ok(21), 42)
        self.assertEqual(ok.__name__, "ok")
        self.assertEqual(ok.__doc__, "docstring")


if __name__ == "__main__":
    unittest.main()
