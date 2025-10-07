"""
经济事件监控器测试模块
"""

import unittest
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path

from src.core.economic_event_monitor import (
    EconomicEventMonitor,
    EconomicCalendar,
    EconomicEvent,
    EIAInventoryMonitor,
    EIAInventoryEvent,
    OPECMeetingMonitor,
    OPECMeeting,
    EventImpact,
    EventType
)


class TestEconomicEvent(unittest.TestCase):
    """经济事件测试"""
    
    def test_is_upcoming(self):
        """测试事件是否即将发生"""
        # 30分钟后的事件
        future_time = datetime.utcnow() + timedelta(minutes=20)
        event = EconomicEvent(
            event_id="test1",
            name="NFP",
            country="US",
            event_type=EventType.EMPLOYMENT,
            impact=EventImpact.HIGH,
            scheduled_time=future_time,
            affected_symbols=["EURUSD", "XAUUSD"]
        )
        
        self.assertTrue(event.is_upcoming(minutes_ahead=30))
        self.assertFalse(event.is_upcoming(minutes_ahead=10))
    
    def test_is_past(self):
        """测试事件是否已过去"""
        past_time = datetime.utcnow() - timedelta(hours=1)
        event = EconomicEvent(
            event_id="test2",
            name="CPI",
            country="US",
            event_type=EventType.INFLATION,
            impact=EventImpact.HIGH,
            scheduled_time=past_time,
            affected_symbols=["EURUSD"]
        )
        
        self.assertTrue(event.is_past())
    
    def test_surprise_index(self):
        """测试意外指数计算"""
        event = EconomicEvent(
            event_id="test3",
            name="GDP",
            country="US",
            event_type=EventType.GDP,
            impact=EventImpact.HIGH,
            scheduled_time=datetime.utcnow(),
            actual_value=2.5,
            forecast_value=2.0,
            previous_value=1.8,
            affected_symbols=["EURUSD"]
        )
        
        surprise = event.get_surprise_index()
        self.assertIsNotNone(surprise)
        self.assertAlmostEqual(surprise, 0.25, places=2)  # (2.5-2.0)/2.0 = 0.25


class TestEconomicCalendar(unittest.TestCase):
    """经济日历测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.calendar = EconomicCalendar()
    
    def test_add_event(self):
        """测试添加事件"""
        event = EconomicEvent(
            event_id="test1",
            name="FOMC",
            country="US",
            event_type=EventType.CENTRAL_BANK,
            impact=EventImpact.CRITICAL,
            scheduled_time=datetime.utcnow() + timedelta(hours=2),
            affected_symbols=["EURUSD", "XAUUSD"]
        )
        
        self.calendar.add_event(event)
        self.assertEqual(len(self.calendar.events), 1)
    
    def test_get_upcoming_events(self):
        """测试获取即将发生的事件"""
        # 添加不同时间的事件
        now = datetime.utcnow()
        
        # 2小时后 - 高影响
        event1 = EconomicEvent(
            event_id="e1",
            name="NFP",
            country="US",
            event_type=EventType.EMPLOYMENT,
            impact=EventImpact.HIGH,
            scheduled_time=now + timedelta(hours=2),
            affected_symbols=["EURUSD"]
        )
        
        # 5小时后 - 中等影响
        event2 = EconomicEvent(
            event_id="e2",
            name="Retail Sales",
            country="US",
            event_type=EventType.CONSUMER,
            impact=EventImpact.MEDIUM,
            scheduled_time=now + timedelta(hours=5),
            affected_symbols=["EURUSD"]
        )
        
        # 30小时后 - 超出范围
        event3 = EconomicEvent(
            event_id="e3",
            name="CPI",
            country="US",
            event_type=EventType.INFLATION,
            impact=EventImpact.HIGH,
            scheduled_time=now + timedelta(hours=30),
            affected_symbols=["EURUSD"]
        )
        
        self.calendar.add_event(event1)
        self.calendar.add_event(event2)
        self.calendar.add_event(event3)
        
        # 获取24小时内的中等及以上影响事件
        upcoming = self.calendar.get_upcoming_events(
            hours_ahead=24,
            min_impact=EventImpact.MEDIUM
        )
        
        self.assertEqual(len(upcoming), 2)
        self.assertEqual(upcoming[0].event_id, "e1")
        self.assertEqual(upcoming[1].event_id, "e2")
    
    def test_get_events_for_symbol(self):
        """测试获取特定品种的事件"""
        now = datetime.utcnow()
        
        event1 = EconomicEvent(
            event_id="e1",
            name="NFP",
            country="US",
            event_type=EventType.EMPLOYMENT,
            impact=EventImpact.HIGH,
            scheduled_time=now + timedelta(hours=2),
            affected_symbols=["EURUSD", "XAUUSD"]
        )
        
        event2 = EconomicEvent(
            event_id="e2",
            name="EIA Inventory",
            country="US",
            event_type=EventType.ENERGY,
            impact=EventImpact.HIGH,
            scheduled_time=now + timedelta(hours=3),
            affected_symbols=["USOIL"]
        )
        
        self.calendar.add_event(event1)
        self.calendar.add_event(event2)
        
        # 获取EURUSD相关事件
        eurusd_events = self.calendar.get_events_for_symbol("EURUSD")
        self.assertEqual(len(eurusd_events), 1)
        self.assertEqual(eurusd_events[0].event_id, "e1")
        
        # 获取USOIL相关事件
        usoil_events = self.calendar.get_events_for_symbol("USOIL")
        self.assertEqual(len(usoil_events), 1)
        self.assertEqual(usoil_events[0].event_id, "e2")
    
    def test_should_avoid_trading(self):
        """测试是否应该避免交易"""
        now = datetime.utcnow()
        
        # 添加20分钟后的高影响事件
        event = EconomicEvent(
            event_id="e1",
            name="FOMC",
            country="US",
            event_type=EventType.CENTRAL_BANK,
            impact=EventImpact.CRITICAL,
            scheduled_time=now + timedelta(minutes=20),
            affected_symbols=["EURUSD"]
        )
        
        self.calendar.add_event(event)
        
        # 应该避免交易
        should_avoid, related_event = self.calendar.should_avoid_trading(
            "EURUSD",
            minutes_before=30,
            minutes_after=15
        )
        
        self.assertTrue(should_avoid)
        self.assertIsNotNone(related_event)
        self.assertEqual(related_event.event_id, "e1")
    
    def test_save_and_load(self):
        """测试保存和加载事件"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name
        
        try:
            # 添加事件
            event = EconomicEvent(
                event_id="e1",
                name="NFP",
                country="US",
                event_type=EventType.EMPLOYMENT,
                impact=EventImpact.HIGH,
                scheduled_time=datetime(2024, 6, 7, 12, 30),
                forecast_value=180000.0,
                previous_value=175000.0,
                affected_symbols=["EURUSD", "XAUUSD"]
            )
            
            self.calendar.add_event(event)
            
            # 保存
            success = self.calendar.save_to_file(temp_file)
            self.assertTrue(success)
            
            # 加载
            new_calendar = EconomicCalendar(temp_file)
            self.assertEqual(len(new_calendar.events), 1)
            
            loaded_event = new_calendar.events[0]
            self.assertEqual(loaded_event.event_id, "e1")
            self.assertEqual(loaded_event.name, "NFP")
            self.assertEqual(loaded_event.impact, EventImpact.HIGH)
            
        finally:
            Path(temp_file).unlink(missing_ok=True)


class TestEIAInventoryMonitor(unittest.TestCase):
    """EIA库存监控器测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.monitor = EIAInventoryMonitor()
    
    def test_get_next_release_time(self):
        """测试获取下一次发布时间"""
        next_release = self.monitor.get_next_release_time()
        
        # 应该是周三
        self.assertEqual(next_release.weekday(), 2)
        
        # 应该是15:30
        self.assertEqual(next_release.hour, 15)
        self.assertEqual(next_release.minute, 30)
        
        # 应该在未来
        self.assertGreater(next_release, datetime.utcnow())
    
    def test_add_inventory_data(self):
        """测试添加库存数据"""
        event = EIAInventoryEvent(
            release_date=datetime.utcnow(),
            crude_oil_change=2.5,
            forecast_crude=1.0,
            previous_crude=0.5
        )
        
        self.monitor.add_inventory_data(event)
        self.assertEqual(len(self.monitor.inventory_events), 1)
    
    def test_get_latest_data(self):
        """测试获取最新数据"""
        # 添加多个数据点
        event1 = EIAInventoryEvent(
            release_date=datetime.utcnow() - timedelta(weeks=2),
            crude_oil_change=1.0
        )
        event2 = EIAInventoryEvent(
            release_date=datetime.utcnow() - timedelta(weeks=1),
            crude_oil_change=2.0
        )
        event3 = EIAInventoryEvent(
            release_date=datetime.utcnow(),
            crude_oil_change=3.0
        )
        
        self.monitor.add_inventory_data(event1)
        self.monitor.add_inventory_data(event2)
        self.monitor.add_inventory_data(event3)
        
        latest = self.monitor.get_latest_data()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.crude_oil_change, 3.0)
    
    def test_analyze_trend(self):
        """测试分析库存趋势"""
        # 添加4周的数据
        for i in range(4):
            event = EIAInventoryEvent(
                release_date=datetime.utcnow() - timedelta(weeks=i),
                crude_oil_change=1.0 + i * 0.5  # 递增趋势
            )
            self.monitor.add_inventory_data(event)
        
        trend = self.monitor.analyze_trend(weeks=4)
        
        self.assertIn('average_change', trend)
        self.assertIn('trend', trend)
        self.assertEqual(trend['trend'], 'building')  # 正值表示库存增加
        self.assertEqual(trend['weeks_analyzed'], 4)


class TestOPECMeetingMonitor(unittest.TestCase):
    """OPEC会议监控器测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.monitor = OPECMeetingMonitor()
    
    def test_add_meeting(self):
        """测试添加会议"""
        meeting = OPECMeeting(
            meeting_date=datetime.utcnow() + timedelta(days=30),
            meeting_type="regular"
        )
        
        initial_count = len(self.monitor.meetings)
        self.monitor.add_meeting(meeting)
        self.assertEqual(len(self.monitor.meetings), initial_count + 1)
    
    def test_get_next_meeting(self):
        """测试获取下一次会议"""
        # 清空现有会议
        self.monitor.meetings = []
        
        # 添加过去和未来的会议
        past_meeting = OPECMeeting(
            meeting_date=datetime.utcnow() - timedelta(days=30),
            meeting_type="regular"
        )
        future_meeting1 = OPECMeeting(
            meeting_date=datetime.utcnow() + timedelta(days=30),
            meeting_type="regular"
        )
        future_meeting2 = OPECMeeting(
            meeting_date=datetime.utcnow() + timedelta(days=60),
            meeting_type="regular"
        )
        
        self.monitor.add_meeting(past_meeting)
        self.monitor.add_meeting(future_meeting1)
        self.monitor.add_meeting(future_meeting2)
        
        next_meeting = self.monitor.get_next_meeting()
        self.assertIsNotNone(next_meeting)
        self.assertEqual(
            next_meeting.meeting_date.date(),
            future_meeting1.meeting_date.date()
        )
    
    def test_update_meeting_decision(self):
        """测试更新会议决议"""
        meeting_date = datetime.utcnow() + timedelta(days=30)
        meeting = OPECMeeting(
            meeting_date=meeting_date,
            meeting_type="regular"
        )
        
        self.monitor.add_meeting(meeting)
        
        # 更新决议
        success = self.monitor.update_meeting_decision(
            meeting_date,
            "Reduce production by 1M bpd",
            -1.0
        )
        
        self.assertTrue(success)
        
        # 验证更新
        updated_meeting = self.monitor.get_next_meeting()
        self.assertEqual(updated_meeting.actual_decision, "Reduce production by 1M bpd")
        self.assertEqual(updated_meeting.production_change, -1.0)
        self.assertTrue(updated_meeting.is_production_cut())


class TestEconomicEventMonitor(unittest.TestCase):
    """经济事件监控器集成测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.monitor = EconomicEventMonitor()
    
    def test_check_trading_restrictions_calendar(self):
        """测试基于日历的交易限制"""
        # 添加即将发生的高影响事件
        event = EconomicEvent(
            event_id="e1",
            name="NFP",
            country="US",
            event_type=EventType.EMPLOYMENT,
            impact=EventImpact.CRITICAL,
            scheduled_time=datetime.utcnow() + timedelta(minutes=20),
            affected_symbols=["EURUSD"]
        )
        
        self.monitor.calendar.add_event(event)
        
        # 检查限制
        should_restrict, reason = self.monitor.check_trading_restrictions("EURUSD")
        self.assertTrue(should_restrict)
        self.assertIn("NFP", reason)
    
    def test_get_event_driven_adjustments(self):
        """测试获取事件驱动的调整建议"""
        # 添加高影响事件
        event = EconomicEvent(
            event_id="e1",
            name="FOMC",
            country="US",
            event_type=EventType.CENTRAL_BANK,
            impact=EventImpact.CRITICAL,
            scheduled_time=datetime.utcnow() + timedelta(hours=2),
            affected_symbols=["EURUSD", "XAUUSD"]
        )
        
        self.monitor.calendar.add_event(event)
        
        # 获取调整建议
        adjustments = self.monitor.get_event_driven_adjustments("EURUSD")
        
        self.assertTrue(adjustments['reduce_position'])
        self.assertTrue(adjustments['increase_stop_loss'])
        self.assertEqual(adjustments['risk_multiplier'], 0.5)
        self.assertGreater(len(adjustments['reasons']), 0)
    
    def test_get_market_sentiment_from_events(self):
        """测试从事件分析市场情绪"""
        # 添加EIA库存数据（库存减少，看涨）
        eia_event = EIAInventoryEvent(
            release_date=datetime.utcnow(),
            crude_oil_change=-2.0,  # 库存减少
            forecast_crude=-1.0
        )
        self.monitor.eia_monitor.add_inventory_data(eia_event)
        
        sentiment = self.monitor.get_market_sentiment_from_events()
        
        # 库存减少应该是看涨
        if 'USOIL' in sentiment:
            self.assertEqual(sentiment['USOIL'], 'bullish')
    
    def test_integration_all_monitors(self):
        """测试所有监控器的集成"""
        # 验证所有监控器都已初始化
        self.assertIsNotNone(self.monitor.calendar)
        self.assertIsNotNone(self.monitor.eia_monitor)
        self.assertIsNotNone(self.monitor.opec_monitor)
        
        # 测试对USOIL的综合检查
        should_restrict, reason = self.monitor.check_trading_restrictions("USOIL")
        
        # 即使没有限制，也应该返回有效结果
        self.assertIsInstance(should_restrict, bool)
        self.assertIsInstance(reason, str)


if __name__ == '__main__':
    unittest.main()
