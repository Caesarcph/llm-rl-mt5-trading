"""
经济事件监控模块
监控经济日历、EIA库存、OPEC会议等重要事件，并提供事件驱动的策略调整
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class EventImpact(Enum):
    """事件影响级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventType(Enum):
    """事件类型"""
    INTEREST_RATE = "interest_rate"
    EMPLOYMENT = "employment"
    INFLATION = "inflation"
    GDP = "gdp"
    MANUFACTURING = "manufacturing"
    CONSUMER = "consumer"
    CENTRAL_BANK = "central_bank"
    ENERGY = "energy"
    GEOPOLITICAL = "geopolitical"
    OTHER = "other"


@dataclass
class EconomicEvent:
    """经济事件数据类"""
    event_id: str
    name: str
    country: str
    event_type: EventType
    impact: EventImpact
    scheduled_time: datetime
    actual_value: Optional[float] = None
    forecast_value: Optional[float] = None
    previous_value: Optional[float] = None
    affected_symbols: List[str] = field(default_factory=list)
    description: str = ""
    
    def is_upcoming(self, minutes_ahead: int = 30) -> bool:
        """检查事件是否即将发生"""
        now = datetime.utcnow()
        time_diff = (self.scheduled_time - now).total_seconds() / 60
        return 0 <= time_diff <= minutes_ahead
    
    def is_past(self) -> bool:
        """检查事件是否已经发生"""
        return datetime.utcnow() > self.scheduled_time
    
    def get_surprise_index(self) -> Optional[float]:
        """计算事件的意外指数（实际值与预期值的偏差）"""
        if self.actual_value is None or self.forecast_value is None:
            return None
        
        if self.forecast_value == 0:
            return None
        
        return (self.actual_value - self.forecast_value) / abs(self.forecast_value)


@dataclass
class EIAInventoryEvent:
    """EIA库存数据事件"""
    release_date: datetime
    crude_oil_change: Optional[float] = None  # 百万桶
    gasoline_change: Optional[float] = None
    distillate_change: Optional[float] = None
    forecast_crude: Optional[float] = None
    previous_crude: Optional[float] = None
    
    def get_surprise(self) -> Optional[float]:
        """获取库存数据的意外程度"""
        if self.crude_oil_change is None or self.forecast_crude is None:
            return None
        return self.crude_oil_change - self.forecast_crude


@dataclass
class OPECMeeting:
    """OPEC会议事件"""
    meeting_date: datetime
    meeting_type: str  # "regular", "extraordinary"
    expected_decision: str = ""
    actual_decision: str = ""
    production_change: Optional[float] = None  # 百万桶/日
    
    def is_production_cut(self) -> bool:
        """是否为减产决议"""
        return self.production_change is not None and self.production_change < 0
    
    def is_production_increase(self) -> bool:
        """是否为增产决议"""
        return self.production_change is not None and self.production_change > 0


class EconomicCalendar:
    """经济日历管理器"""
    
    def __init__(self, data_file: Optional[str] = None):
        """
        初始化经济日历
        
        Args:
            data_file: 日历数据文件路径
        """
        self.events: List[EconomicEvent] = []
        self.data_file = data_file
        
        if data_file and Path(data_file).exists():
            self.load_from_file(data_file)
        else:
            self._initialize_default_events()
    
    def _initialize_default_events(self):
        """初始化默认的重要经济事件"""
        # 这里添加一些常见的重要经济事件模板
        logger.info("初始化默认经济事件日历")
    
    def add_event(self, event: EconomicEvent) -> None:
        """添加经济事件"""
        self.events.append(event)
        logger.debug(f"添加经济事件: {event.name} at {event.scheduled_time}")
    
    def get_upcoming_events(self, hours_ahead: int = 24, 
                          min_impact: EventImpact = EventImpact.MEDIUM) -> List[EconomicEvent]:
        """
        获取即将发生的重要事件
        
        Args:
            hours_ahead: 未来多少小时内的事件
            min_impact: 最小影响级别
            
        Returns:
            即将发生的事件列表
        """
        now = datetime.utcnow()
        cutoff_time = now + timedelta(hours=hours_ahead)
        
        impact_levels = {
            EventImpact.LOW: 0,
            EventImpact.MEDIUM: 1,
            EventImpact.HIGH: 2,
            EventImpact.CRITICAL: 3
        }
        
        min_level = impact_levels[min_impact]
        
        upcoming = [
            event for event in self.events
            if now <= event.scheduled_time <= cutoff_time
            and impact_levels[event.impact] >= min_level
        ]
        
        return sorted(upcoming, key=lambda e: e.scheduled_time)
    
    def get_events_for_symbol(self, symbol: str, hours_ahead: int = 24) -> List[EconomicEvent]:
        """
        获取影响特定品种的事件
        
        Args:
            symbol: 品种代码
            hours_ahead: 未来多少小时内的事件
            
        Returns:
            相关事件列表
        """
        upcoming = self.get_upcoming_events(hours_ahead, EventImpact.LOW)
        return [event for event in upcoming if symbol in event.affected_symbols]
    
    def should_avoid_trading(self, symbol: str, minutes_before: int = 30,
                           minutes_after: int = 15) -> Tuple[bool, Optional[EconomicEvent]]:
        """
        检查是否应该避免交易
        
        Args:
            symbol: 品种代码
            minutes_before: 事件前多少分钟避免交易
            minutes_after: 事件后多少分钟避免交易
            
        Returns:
            (是否应该避免, 相关事件)
        """
        now = datetime.utcnow()
        
        for event in self.events:
            if symbol not in event.affected_symbols:
                continue
            
            if event.impact not in [EventImpact.HIGH, EventImpact.CRITICAL]:
                continue
            
            time_diff = (event.scheduled_time - now).total_seconds() / 60
            
            # 在事件前后的避免交易时段内
            if -minutes_after <= time_diff <= minutes_before:
                return True, event
        
        return False, None
    
    def load_from_file(self, file_path: str) -> None:
        """从文件加载事件数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.events = []
            for event_data in data.get('events', []):
                event = EconomicEvent(
                    event_id=event_data['event_id'],
                    name=event_data['name'],
                    country=event_data['country'],
                    event_type=EventType(event_data['event_type']),
                    impact=EventImpact(event_data['impact']),
                    scheduled_time=datetime.fromisoformat(event_data['scheduled_time']),
                    actual_value=event_data.get('actual_value'),
                    forecast_value=event_data.get('forecast_value'),
                    previous_value=event_data.get('previous_value'),
                    affected_symbols=event_data.get('affected_symbols', []),
                    description=event_data.get('description', '')
                )
                self.events.append(event)
            
            logger.info(f"从文件加载了 {len(self.events)} 个经济事件")
        except Exception as e:
            logger.error(f"加载经济事件文件失败: {e}")
    
    def save_to_file(self, file_path: str) -> bool:
        """保存事件数据到文件"""
        try:
            data = {
                'events': [
                    {
                        'event_id': event.event_id,
                        'name': event.name,
                        'country': event.country,
                        'event_type': event.event_type.value,
                        'impact': event.impact.value,
                        'scheduled_time': event.scheduled_time.isoformat(),
                        'actual_value': event.actual_value,
                        'forecast_value': event.forecast_value,
                        'previous_value': event.previous_value,
                        'affected_symbols': event.affected_symbols,
                        'description': event.description
                    }
                    for event in self.events
                ]
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"保存了 {len(self.events)} 个经济事件到文件")
            return True
        except Exception as e:
            logger.error(f"保存经济事件文件失败: {e}")
            return False


class EIAInventoryMonitor:
    """EIA库存数据监控器"""
    
    def __init__(self):
        """初始化EIA监控器"""
        self.inventory_events: List[EIAInventoryEvent] = []
        self.release_day = 2  # 周三 (0=Monday, 2=Wednesday)
        self.release_time = (15, 30)  # 15:30 UTC
    
    def get_next_release_time(self) -> datetime:
        """获取下一次EIA发布时间"""
        now = datetime.utcnow()
        
        # 计算到下一个周三的天数
        days_ahead = self.release_day - now.weekday()
        if days_ahead <= 0:  # 如果今天是周三或之后
            days_ahead += 7
        
        next_release = now + timedelta(days=days_ahead)
        next_release = next_release.replace(
            hour=self.release_time[0],
            minute=self.release_time[1],
            second=0,
            microsecond=0
        )
        
        return next_release
    
    def is_near_release(self, minutes_before: int = 30, 
                       minutes_after: int = 15) -> Tuple[bool, Optional[datetime]]:
        """
        检查是否临近EIA发布时间
        
        Args:
            minutes_before: 发布前多少分钟
            minutes_after: 发布后多少分钟
            
        Returns:
            (是否临近, 发布时间)
        """
        now = datetime.utcnow()
        next_release = self.get_next_release_time()
        
        time_diff = (next_release - now).total_seconds() / 60
        
        if -minutes_after <= time_diff <= minutes_before:
            return True, next_release
        
        return False, None
    
    def add_inventory_data(self, event: EIAInventoryEvent) -> None:
        """添加库存数据"""
        self.inventory_events.append(event)
        logger.info(f"添加EIA库存数据: {event.release_date}")
    
    def get_latest_data(self) -> Optional[EIAInventoryEvent]:
        """获取最新的库存数据"""
        if not self.inventory_events:
            return None
        
        return max(self.inventory_events, key=lambda e: e.release_date)
    
    def analyze_trend(self, weeks: int = 4) -> Dict[str, float]:
        """
        分析库存趋势
        
        Args:
            weeks: 分析最近几周的数据
            
        Returns:
            趋势分析结果
        """
        if not self.inventory_events:
            return {}
        
        cutoff_date = datetime.utcnow() - timedelta(weeks=weeks)
        recent_events = [
            e for e in self.inventory_events
            if e.release_date >= cutoff_date and e.crude_oil_change is not None
        ]
        
        if not recent_events:
            return {}
        
        crude_changes = [e.crude_oil_change for e in recent_events]
        avg_change = sum(crude_changes) / len(crude_changes)
        
        return {
            'average_change': avg_change,
            'total_change': sum(crude_changes),
            'weeks_analyzed': len(recent_events),
            'trend': 'building' if avg_change > 0 else 'drawing'
        }


class OPECMeetingMonitor:
    """OPEC会议监控器"""
    
    def __init__(self):
        """初始化OPEC会议监控器"""
        self.meetings: List[OPECMeeting] = []
        self._load_scheduled_meetings()
    
    def _load_scheduled_meetings(self):
        """加载已计划的OPEC会议"""
        # 2024年OPEC+会议日程（示例）
        scheduled_dates = [
            "2024-06-01",
            "2024-12-01"
        ]
        
        for date_str in scheduled_dates:
            meeting_date = datetime.strptime(date_str, "%Y-%m-%d")
            if meeting_date > datetime.utcnow():
                self.meetings.append(OPECMeeting(
                    meeting_date=meeting_date,
                    meeting_type="regular"
                ))
        
        logger.info(f"加载了 {len(self.meetings)} 个OPEC会议")
    
    def add_meeting(self, meeting: OPECMeeting) -> None:
        """添加OPEC会议"""
        self.meetings.append(meeting)
        self.meetings.sort(key=lambda m: m.meeting_date)
        logger.info(f"添加OPEC会议: {meeting.meeting_date}")
    
    def get_next_meeting(self) -> Optional[OPECMeeting]:
        """获取下一次会议"""
        now = datetime.utcnow()
        future_meetings = [m for m in self.meetings if m.meeting_date > now]
        
        if not future_meetings:
            return None
        
        return min(future_meetings, key=lambda m: m.meeting_date)
    
    def is_near_meeting(self, hours_before: int = 2,
                       hours_after: int = 1) -> Tuple[bool, Optional[OPECMeeting]]:
        """
        检查是否临近OPEC会议
        
        Args:
            hours_before: 会议前多少小时
            hours_after: 会议后多少小时
            
        Returns:
            (是否临近, 会议对象)
        """
        now = datetime.utcnow()
        
        for meeting in self.meetings:
            time_diff = (meeting.meeting_date - now).total_seconds() / 3600
            
            if -hours_after <= time_diff <= hours_before:
                return True, meeting
        
        return False, None
    
    def update_meeting_decision(self, meeting_date: datetime,
                               decision: str, production_change: float) -> bool:
        """
        更新会议决议
        
        Args:
            meeting_date: 会议日期
            decision: 决议内容
            production_change: 产量变化
            
        Returns:
            是否更新成功
        """
        for meeting in self.meetings:
            if meeting.meeting_date.date() == meeting_date.date():
                meeting.actual_decision = decision
                meeting.production_change = production_change
                logger.info(f"更新OPEC会议决议: {decision}")
                return True
        
        return False


class EconomicEventMonitor:
    """经济事件监控器（整合所有事件监控）"""
    
    def __init__(self, calendar_file: Optional[str] = None):
        """
        初始化经济事件监控器
        
        Args:
            calendar_file: 经济日历数据文件
        """
        self.calendar = EconomicCalendar(calendar_file)
        self.eia_monitor = EIAInventoryMonitor()
        self.opec_monitor = OPECMeetingMonitor()
        
        logger.info("经济事件监控器初始化完成")
    
    def check_trading_restrictions(self, symbol: str) -> Tuple[bool, str]:
        """
        检查品种是否有交易限制
        
        Args:
            symbol: 品种代码
            
        Returns:
            (是否应该限制交易, 限制原因)
        """
        # 检查经济日历事件
        should_avoid, event = self.calendar.should_avoid_trading(symbol)
        if should_avoid:
            return True, f"重要经济事件临近: {event.name}"
        
        # 检查EIA库存数据（仅对原油相关品种）
        if symbol in ['USOIL', 'UKOIL', 'CL']:
            is_near, release_time = self.eia_monitor.is_near_release()
            if is_near:
                return True, f"EIA库存数据发布临近: {release_time}"
        
        # 检查OPEC会议（仅对原油相关品种）
        if symbol in ['USOIL', 'UKOIL', 'CL']:
            is_near, meeting = self.opec_monitor.is_near_meeting()
            if is_near:
                return True, f"OPEC会议临近: {meeting.meeting_date}"
        
        return False, ""
    
    def get_event_driven_adjustments(self, symbol: str) -> Dict[str, any]:
        """
        获取基于事件的策略调整建议
        
        Args:
            symbol: 品种代码
            
        Returns:
            调整建议字典
        """
        adjustments = {
            'reduce_position': False,
            'increase_stop_loss': False,
            'avoid_new_trades': False,
            'risk_multiplier': 1.0,
            'reasons': []
        }
        
        # 检查即将发生的高影响事件
        upcoming_events = self.calendar.get_events_for_symbol(symbol, hours_ahead=4)
        high_impact_events = [
            e for e in upcoming_events
            if e.impact in [EventImpact.HIGH, EventImpact.CRITICAL]
        ]
        
        if high_impact_events:
            adjustments['reduce_position'] = True
            adjustments['increase_stop_loss'] = True
            adjustments['risk_multiplier'] = 0.5
            adjustments['reasons'].append(
                f"检测到 {len(high_impact_events)} 个高影响事件"
            )
        
        # 检查EIA和OPEC事件
        should_restrict, reason = self.check_trading_restrictions(symbol)
        if should_restrict:
            adjustments['avoid_new_trades'] = True
            adjustments['reasons'].append(reason)
        
        return adjustments
    
    def get_market_sentiment_from_events(self) -> Dict[str, str]:
        """
        从事件中分析市场情绪
        
        Returns:
            各品种的市场情绪
        """
        sentiment = {}
        
        # 分析EIA库存趋势对原油的影响
        eia_trend = self.eia_monitor.analyze_trend()
        if eia_trend:
            if eia_trend['trend'] == 'building':
                sentiment['USOIL'] = 'bearish'  # 库存增加，看跌
            else:
                sentiment['USOIL'] = 'bullish'  # 库存减少，看涨
        
        # 分析OPEC决议对原油的影响
        next_meeting = self.opec_monitor.get_next_meeting()
        if next_meeting and next_meeting.actual_decision:
            if next_meeting.is_production_cut():
                sentiment['USOIL'] = 'bullish'  # 减产，看涨
            elif next_meeting.is_production_increase():
                sentiment['USOIL'] = 'bearish'  # 增产，看跌
        
        return sentiment
