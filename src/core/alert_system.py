"""
告警通知系统
提供Telegram Bot和邮件通知、分级告警机制和自定义告警规则配置
"""

import smtplib
import asyncio
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import json
from pathlib import Path

from .logging import get_logger, LoggerMixin
from .exceptions import TradingSystemException


class AlertLevel(Enum):
    """告警级别枚举"""
    INFO = "info"           # 信息
    WARNING = "warning"     # 警告
    CRITICAL = "critical"   # 严重


class AlertChannel(Enum):
    """告警渠道枚举"""
    TELEGRAM = "telegram"
    EMAIL = "email"
    LOG = "log"
    ALL = "all"


@dataclass
class Alert:
    """告警消息"""
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'level': self.level.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'metadata': self.metadata
        }
    
    def format_message(self) -> str:
        """格式化消息"""
        emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.CRITICAL: "🚨"
        }
        
        msg = f"{emoji.get(self.level, '')} {self.level.value.upper()}\n"
        msg += f"📌 {self.title}\n"
        msg += f"📝 {self.message}\n"
        msg += f"🕐 {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        msg += f"📍 Source: {self.source}"
        
        if self.metadata:
            msg += f"\n📊 Details: {json.dumps(self.metadata, indent=2)}"
        
        return msg


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    level: AlertLevel
    message_template: str
    channels: List[AlertChannel]
    cooldown_minutes: int = 5  # 冷却时间，避免重复告警
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    
    def can_trigger(self) -> bool:
        """检查是否可以触发"""
        if not self.enabled:
            return False
        
        if self.last_triggered is None:
            return True
        
        cooldown_period = timedelta(minutes=self.cooldown_minutes)
        return datetime.now() - self.last_triggered > cooldown_period
    
    def evaluate(self, context: Dict[str, Any]) -> Optional[Alert]:
        """评估规则并生成告警"""
        if not self.can_trigger():
            return None
        
        try:
            if self.condition(context):
                self.last_triggered = datetime.now()
                message = self.message_template.format(**context)
                return Alert(
                    level=self.level,
                    title=self.name,
                    message=message,
                    source="rule_engine",
                    metadata=context
                )
        except Exception as e:
            # 规则评估失败，记录但不抛出异常
            pass
        
        return None


class TelegramNotifier(LoggerMixin):
    """Telegram通知器"""
    
    def __init__(self, bot_token: str, chat_id: str):
        super().__init__()
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}"
        self.enabled = bool(bot_token and chat_id)
        
        if not self.enabled:
            self.log_warning("Telegram notifier disabled: missing bot_token or chat_id")
    
    async def send_message_async(self, message: str) -> bool:
        """异步发送消息"""
        if not self.enabled:
            return False
        
        try:
            import aiohttp
            
            url = f"{self.api_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        self.log_info("Telegram message sent successfully")
                        return True
                    else:
                        error_text = await response.text()
                        self.log_error(f"Failed to send Telegram message: {error_text}")
                        return False
        
        except Exception as e:
            self.log_error(f"Telegram notification error: {e}")
            return False
    
    def send_message(self, message: str) -> bool:
        """同步发送消息"""
        if not self.enabled:
            return False
        
        try:
            # 在新的事件循环中运行异步函数
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.send_message_async(message))
            loop.close()
            return result
        except Exception as e:
            self.log_error(f"Failed to send Telegram message: {e}")
            return False


class EmailNotifier(LoggerMixin):
    """邮件通知器"""
    
    def __init__(self, smtp_host: str, smtp_port: int, username: str, 
                 password: str, from_addr: str, to_addrs: List[str]):
        super().__init__()
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs
        self.enabled = bool(smtp_host and username and password and to_addrs)
        
        if not self.enabled:
            self.log_warning("Email notifier disabled: missing configuration")
    
    def send_email(self, subject: str, body: str, html: bool = False) -> bool:
        """发送邮件"""
        if not self.enabled:
            return False
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_addr
            msg['To'] = ', '.join(self.to_addrs)
            
            if html:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            self.log_info(f"Email sent successfully to {', '.join(self.to_addrs)}")
            return True
        
        except Exception as e:
            self.log_error(f"Email notification error: {e}")
            return False


class AlertSystem(LoggerMixin):
    """告警系统"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 初始化通知器
        telegram_config = config.get('telegram', {})
        self.telegram_notifier = TelegramNotifier(
            bot_token=telegram_config.get('bot_token', ''),
            chat_id=telegram_config.get('chat_id', '')
        )
        
        email_config = config.get('email', {})
        self.email_notifier = EmailNotifier(
            smtp_host=email_config.get('smtp_host', ''),
            smtp_port=email_config.get('smtp_port', 587),
            username=email_config.get('username', ''),
            password=email_config.get('password', ''),
            from_addr=email_config.get('from_addr', ''),
            to_addrs=email_config.get('to_addrs', [])
        )
        
        # 告警规则
        self.rules: Dict[str, AlertRule] = {}
        
        # 告警历史
        self.alert_history: List[Alert] = []
        self.max_history_size = config.get('max_history_size', 1000)
        
        # 注册默认规则
        self._register_default_rules()
        
        self.log_info("Alert system initialized")
    
    def _register_default_rules(self):
        """注册默认告警规则"""
        # 账户余额告警
        self.register_rule(AlertRule(
            name="Low Account Balance",
            condition=lambda ctx: ctx.get('balance', float('inf')) < ctx.get('min_balance', 1000),
            level=AlertLevel.WARNING,
            message_template="Account balance is low: ${balance:.2f}",
            channels=[AlertChannel.TELEGRAM, AlertChannel.EMAIL],
            cooldown_minutes=60
        ))
        
        # 日回撤告警
        self.register_rule(AlertRule(
            name="Daily Drawdown Exceeded",
            condition=lambda ctx: ctx.get('daily_drawdown', 0) > ctx.get('max_daily_drawdown', 0.05),
            level=AlertLevel.CRITICAL,
            message_template="Daily drawdown exceeded: {daily_drawdown:.2%} > {max_daily_drawdown:.2%}",
            channels=[AlertChannel.ALL],
            cooldown_minutes=30
        ))
        
        # 连续亏损告警
        self.register_rule(AlertRule(
            name="Consecutive Losses",
            condition=lambda ctx: ctx.get('consecutive_losses', 0) >= 3,
            level=AlertLevel.WARNING,
            message_template="Consecutive losses detected: {consecutive_losses} trades",
            channels=[AlertChannel.TELEGRAM],
            cooldown_minutes=120
        ))
        
        # 系统错误告警
        self.register_rule(AlertRule(
            name="System Error",
            condition=lambda ctx: ctx.get('error_count', 0) > 5,
            level=AlertLevel.CRITICAL,
            message_template="Multiple system errors detected: {error_count} errors in last hour",
            channels=[AlertChannel.ALL],
            cooldown_minutes=15
        ))
        
        # MT5连接断开告警
        self.register_rule(AlertRule(
            name="MT5 Connection Lost",
            condition=lambda ctx: not ctx.get('mt5_connected', True),
            level=AlertLevel.CRITICAL,
            message_template="MT5 connection lost. Trading halted.",
            channels=[AlertChannel.ALL],
            cooldown_minutes=5
        ))
    
    def register_rule(self, rule: AlertRule):
        """注册告警规则"""
        self.rules[rule.name] = rule
        self.log_info(f"Registered alert rule: {rule.name}")
    
    def unregister_rule(self, rule_name: str):
        """注销告警规则"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            self.log_info(f"Unregistered alert rule: {rule_name}")
    
    def enable_rule(self, rule_name: str):
        """启用告警规则"""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = True
            self.log_info(f"Enabled alert rule: {rule_name}")
    
    def disable_rule(self, rule_name: str):
        """禁用告警规则"""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = False
            self.log_info(f"Disabled alert rule: {rule_name}")
    
    def evaluate_rules(self, context: Dict[str, Any]) -> List[Alert]:
        """评估所有规则"""
        alerts = []
        
        for rule in self.rules.values():
            alert = rule.evaluate(context)
            if alert:
                alerts.append(alert)
        
        return alerts
    
    def send_alert(self, alert: Alert, channels: Optional[List[AlertChannel]] = None):
        """发送告警"""
        if channels is None:
            channels = [AlertChannel.ALL]
        
        # 记录到历史
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history_size:
            self.alert_history.pop(0)
        
        # 格式化消息
        formatted_message = alert.format_message()
        
        # 发送到各个渠道
        for channel in channels:
            if channel == AlertChannel.ALL:
                self._send_to_all_channels(alert, formatted_message)
            elif channel == AlertChannel.TELEGRAM:
                self._send_to_telegram(formatted_message)
            elif channel == AlertChannel.EMAIL:
                self._send_to_email(alert, formatted_message)
            elif channel == AlertChannel.LOG:
                self._send_to_log(alert)
    
    def _send_to_all_channels(self, alert: Alert, formatted_message: str):
        """发送到所有渠道"""
        self._send_to_telegram(formatted_message)
        self._send_to_email(alert, formatted_message)
        self._send_to_log(alert)
    
    def _send_to_telegram(self, message: str):
        """发送到Telegram"""
        self.telegram_notifier.send_message(message)
    
    def _send_to_email(self, alert: Alert, message: str):
        """发送到邮件"""
        subject = f"[{alert.level.value.upper()}] {alert.title}"
        self.email_notifier.send_email(subject, message)
    
    def _send_to_log(self, alert: Alert):
        """记录到日志"""
        log_methods = {
            AlertLevel.INFO: self.log_info,
            AlertLevel.WARNING: self.log_warning,
            AlertLevel.CRITICAL: self.log_error
        }
        
        log_method = log_methods.get(alert.level, self.log_info)
        log_method(f"[{alert.title}] {alert.message}")
    
    def send_info(self, title: str, message: str, **metadata):
        """发送信息级别告警"""
        alert = Alert(
            level=AlertLevel.INFO,
            title=title,
            message=message,
            metadata=metadata
        )
        self.send_alert(alert, [AlertChannel.LOG])
    
    def send_warning(self, title: str, message: str, **metadata):
        """发送警告级别告警"""
        alert = Alert(
            level=AlertLevel.WARNING,
            title=title,
            message=message,
            metadata=metadata
        )
        self.send_alert(alert, [AlertChannel.TELEGRAM, AlertChannel.LOG])
    
    def send_critical(self, title: str, message: str, **metadata):
        """发送严重级别告警"""
        alert = Alert(
            level=AlertLevel.CRITICAL,
            title=title,
            message=message,
            metadata=metadata
        )
        self.send_alert(alert, [AlertChannel.ALL])
    
    def get_alert_history(self, level: Optional[AlertLevel] = None, 
                         limit: int = 100) -> List[Alert]:
        """获取告警历史"""
        if level:
            filtered = [a for a in self.alert_history if a.level == level]
        else:
            filtered = self.alert_history
        
        return filtered[-limit:]
    
    def get_alert_stats(self) -> Dict[str, int]:
        """获取告警统计"""
        stats = {
            'total': len(self.alert_history),
            'info': 0,
            'warning': 0,
            'critical': 0
        }
        
        for alert in self.alert_history:
            stats[alert.level.value] += 1
        
        return stats
    
    def clear_history(self):
        """清除告警历史"""
        self.alert_history.clear()
        self.log_info("Alert history cleared")
    
    def save_config(self, config_path: str):
        """保存配置"""
        config_data = {
            'rules': {
                name: {
                    'name': rule.name,
                    'level': rule.level.value,
                    'message_template': rule.message_template,
                    'channels': [ch.value for ch in rule.channels],
                    'cooldown_minutes': rule.cooldown_minutes,
                    'enabled': rule.enabled
                }
                for name, rule in self.rules.items()
            }
        }
        
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        self.log_info(f"Alert configuration saved to {config_path}")
    
    def load_config(self, config_path: str):
        """加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 加载规则配置
            for rule_name, rule_config in config_data.get('rules', {}).items():
                if rule_name in self.rules:
                    rule = self.rules[rule_name]
                    rule.enabled = rule_config.get('enabled', True)
                    rule.cooldown_minutes = rule_config.get('cooldown_minutes', 5)
            
            self.log_info(f"Alert configuration loaded from {config_path}")
        
        except Exception as e:
            self.log_error(f"Failed to load alert configuration: {e}")
