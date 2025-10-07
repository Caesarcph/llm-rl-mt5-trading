"""
核心模块测试
"""

import unittest
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.models import (
    MarketData, Signal, Account, Position, Trade, 
    PositionType, TradeType, RiskMetrics, MarketState, SymbolConfig
)
from src.core.config import ConfigManager, SystemConfig
from src.core.exceptions import (
    DataValidationException, OrderException, RiskLimitExceededException
)


class TestModels(unittest.TestCase):
    """测试数据模型"""
    
    def setUp(self):
        """测试设置"""
        self.sample_ohlcv = pd.DataFrame({
            'time': [datetime.now()],
            'open': [1.1000],
            'high': [1.1010],
            'low': [1.0990],
            'close': [1.1005],
            'volume': [1000]
        })
    
    def test_market_data_creation(self):
        """测试市场数据创建"""
        market_data = MarketData(
            symbol="EURUSD",
            timeframe="H1",
            timestamp=datetime.now(),
            ohlcv=self.sample_ohlcv,
            spread=1.5
        )
        
        self.assertEqual(market_data.symbol, "EURUSD")
        self.assertEqual(market_data.timeframe, "H1")
        self.assertEqual(market_data.spread, 1.5)
    
    def test_market_data_validation(self):
        """测试市场数据验证"""
        with self.assertRaises(ValueError):
            MarketData(
                symbol="EURUSD",
                timeframe="H1", 
                timestamp=datetime.now(),
                ohlcv=pd.DataFrame(),  # 空数据框
                spread=-1.0  # 负点差
            )
    
    def test_signal_creation(self):
        """测试信号创建"""
        signal = Signal(
            strategy_id="test_strategy",
            symbol="EURUSD",
            direction=1,
            strength=0.8,
            entry_price=1.1000,
            sl=1.0950,
            tp=1.1050,
            size=0.1,
            confidence=0.9,
            timestamp=datetime.now()
        )
        
        self.assertEqual(signal.strategy_id, "test_strategy")
        self.assertEqual(signal.direction, 1)
        self.assertEqual(signal.strength, 0.8)
    
    def test_signal_validation(self):
        """测试信号验证"""
        with self.assertRaises(ValueError):
            Signal(
                strategy_id="test",
                symbol="EURUSD",
                direction=2,  # 无效方向
                strength=1.5,  # 无效强度
                entry_price=1.1000,
                sl=1.0950,
                tp=1.1050,
                size=-0.1,  # 无效手数
                confidence=0.9,
                timestamp=datetime.now()
            )
    
    def test_account_methods(self):
        """测试账户方法"""
        account = Account(
            account_id="12345",
            balance=10000.0,
            equity=9800.0,
            margin=500.0,
            free_margin=9300.0,
            margin_level=1960.0,
            currency="USD",
            leverage=100
        )
        
        self.assertTrue(account.can_open_position(1000.0))
        self.assertFalse(account.can_open_position(10000.0))
        self.assertAlmostEqual(account.get_margin_level_percent(), 1960.0, places=5)
    
    def test_position_calculations(self):
        """测试持仓计算"""
        position = Position(
            position_id="pos_001",
            symbol="EURUSD",
            type=PositionType.LONG,
            volume=1.0,
            open_price=1.1000,
            current_price=1.1050
        )
        
        pnl = position.calculate_unrealized_pnl()
        self.assertAlmostEqual(pnl, 0.005, places=5)  # (1.1050 - 1.1000) * 1.0
        
        position.update_current_price(1.0950)
        self.assertAlmostEqual(position.profit, -0.005, places=5)  # (1.0950 - 1.1000) * 1.0
        self.assertFalse(position.is_profitable())


class TestConfig(unittest.TestCase):
    """测试配置管理"""
    
    def setUp(self):
        """测试设置"""
        self.config_manager = ConfigManager("test_config")
    
    def test_default_config_creation(self):
        """测试默认配置创建"""
        config = SystemConfig()
        self.assertTrue(config.validate())
        self.assertEqual(config.risk.max_risk_per_trade, 0.02)
        self.assertEqual(config.trading.default_lot_size, 0.01)
    
    def test_config_validation(self):
        """测试配置验证"""
        config = SystemConfig()
        
        # 测试有效配置
        self.assertTrue(config.validate())
        
        # 测试无效配置
        config.risk.max_risk_per_trade = 0.5  # 50%风险过高
        self.assertFalse(config.validate())
    
    def test_symbol_config(self):
        """测试品种配置"""
        symbol_config = SymbolConfig(
            symbol="EURUSD",
            spread_limit=2.0,
            min_lot=0.01,
            max_lot=10.0,
            lot_step=0.01
        )
        
        # 测试手数验证
        self.assertEqual(symbol_config.validate_lot_size(0.005), 0.01)  # 低于最小值
        self.assertEqual(symbol_config.validate_lot_size(15.0), 10.0)   # 高于最大值
        self.assertEqual(symbol_config.validate_lot_size(0.123), 0.12)  # 调整到步长


class TestExceptions(unittest.TestCase):
    """测试异常处理"""
    
    def test_trading_system_exception(self):
        """测试交易系统异常"""
        exception = DataValidationException(
            "测试异常",
            field_name="test_field",
            field_value="invalid_value"
        )
        
        self.assertEqual(exception.message, "测试异常")
        self.assertEqual(exception.field_name, "test_field")
        self.assertEqual(exception.error_code, "DATA_VALIDATION")
    
    def test_risk_limit_exception(self):
        """测试风险限制异常"""
        exception = RiskLimitExceededException(
            "最大回撤",
            0.15,
            0.10
        )
        
        self.assertEqual(exception.risk_type, "最大回撤")
        self.assertEqual(exception.current_value, 0.15)
        self.assertEqual(exception.limit_value, 0.10)


if __name__ == '__main__':
    unittest.main()