"""
告警系统测试

原文件内容已损坏（字符顺序被打乱，无法解析），按 src/core/alert_system.py 的实际接口重写。
所有通知渠道都不配置凭证，因此不会发出任何网络请求。
"""

import unittest
from datetime import datetime, timedelta

from src.core.alert_system import (
    Alert,
    AlertChannel,
    AlertLevel,
    AlertRule,
    AlertSystem,
    EmailNotifier,
    TelegramNotifier,
)


class TestAlert(unittest.TestCase):
    def test_to_dict_and_format(self):
        alert = Alert(level=AlertLevel.WARNING, title="Test", message="hello", metadata={"x": 1})
        data = alert.to_dict()
        self.assertEqual(data["level"], "warning")
        self.assertEqual(data["title"], "Test")
        self.assertEqual(data["metadata"], {"x": 1})

        text = alert.format_message()
        self.assertIn("WARNING", text)
        self.assertIn("hello", text)
        self.assertIn('"x": 1', text)


class TestAlertRule(unittest.TestCase):
    def make_rule(self, **kwargs):
        defaults = dict(
            name="High DD",
            condition=lambda ctx: ctx["dd"] > 0.1,
            level=AlertLevel.CRITICAL,
            message_template="drawdown {dd:.0%}",
            channels=[AlertChannel.LOG],
            cooldown_minutes=5,
        )
        defaults.update(kwargs)
        return AlertRule(**defaults)

    def test_triggers_and_formats_message(self):
        rule = self.make_rule()
        alert = rule.evaluate({"dd": 0.25})
        self.assertIsNotNone(alert)
        self.assertEqual(alert.level, AlertLevel.CRITICAL)
        self.assertEqual(alert.message, "drawdown 25%")

    def test_no_alert_when_condition_false(self):
        self.assertIsNone(self.make_rule().evaluate({"dd": 0.05}))

    def test_cooldown_blocks_repeat_then_expires(self):
        rule = self.make_rule()
        self.assertIsNotNone(rule.evaluate({"dd": 0.25}))
        self.assertIsNone(rule.evaluate({"dd": 0.25}))  # 冷却期内

        rule.last_triggered = datetime.now() - timedelta(minutes=6)
        self.assertIsNotNone(rule.evaluate({"dd": 0.25}))

    def test_disabled_rule_never_triggers(self):
        self.assertIsNone(self.make_rule(enabled=False).evaluate({"dd": 0.25}))

    def test_broken_condition_is_swallowed(self):
        rule = self.make_rule(condition=lambda ctx: ctx["missing_key"] > 0)
        self.assertIsNone(rule.evaluate({"dd": 0.25}))


class TestNotifiersWithoutCredentials(unittest.TestCase):
    def test_telegram_disabled_without_token(self):
        notifier = TelegramNotifier(bot_token="", chat_id="")
        self.assertFalse(notifier.enabled)
        self.assertFalse(notifier.send_message("hi"))

    def test_email_disabled_without_config(self):
        notifier = EmailNotifier("", 587, "", "", "", [])
        self.assertFalse(notifier.enabled)
        self.assertFalse(notifier.send_email("subject", "body"))


class TestAlertSystem(unittest.TestCase):
    def setUp(self):
        self.system = AlertSystem({"max_history_size": 3})

    def test_default_rules_registered(self):
        self.assertIn("MT5 Connection Lost", self.system.rules)
        self.assertIn("Daily Drawdown Exceeded", self.system.rules)

    def test_evaluate_default_rules(self):
        alerts = self.system.evaluate_rules({"mt5_connected": False})
        self.assertEqual([a.title for a in alerts], ["MT5 Connection Lost"])

    def test_disable_and_unregister_rule(self):
        self.system.disable_rule("MT5 Connection Lost")
        self.assertEqual(self.system.evaluate_rules({"mt5_connected": False}), [])

        self.system.unregister_rule("MT5 Connection Lost")
        self.assertNotIn("MT5 Connection Lost", self.system.rules)

    def test_history_is_bounded_and_counted(self):
        for i in range(5):
            self.system.send_info(f"info {i}", "msg")
        self.system.send_critical("boom", "msg")

        history = self.system.get_alert_history()
        self.assertEqual(len(history), 3)
        self.assertEqual(history[-1].title, "boom")

        stats = self.system.get_alert_stats()
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["critical"], 1)
        self.assertEqual(len(self.system.get_alert_history(level=AlertLevel.CRITICAL)), 1)

        self.system.clear_history()
        self.assertEqual(self.system.get_alert_stats()["total"], 0)


if __name__ == "__main__":
    unittest.main()
