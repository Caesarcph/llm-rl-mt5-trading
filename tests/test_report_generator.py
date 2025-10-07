"""
报告生成器测试
测试每日、周、月报告生成，策略分析报告，风险分析报告和合规检查报告
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import tempfile
import json
from pathlib import Path

from src.utils.report_generator import (
    ReportGenerator, ReportType, ReportFormat, TradingReport,
    ComplianceCheckResult, MaxRiskPerTradeRule, MaxDailyDrawdownRule
)
from src.strategies.performance_tracker import (
    PerformanceTracker, PerformancePeriod, StrategyPerformanceMetrics
)
from src.agents.risk_manager import RiskManagerAgent, RiskConfig
from src.core.models import Trade, Position, Account, Signal, PositionType, TradeType


class TestReportGenerator(unittest.TestCase):
    """报告生成器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建性能跟踪器
        self.performance_tracker = PerformanceTracker()
        
        # 创建风险管理器
        self.risk_manager = RiskManagerAgent(RiskConfig())
        
        # 创建报告生成器
        self.report_generator = ReportGenerator(
            performance_tracker=self.performance_tracker,
            risk_manager=self.risk_manager,
            output_dir=self.temp_dir
        )
        
        # 准备测试数据
        self._prepare_test_data()
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _prepare_test_data(self):
        """准备测试数据"""
        # 创建测试交易记录
        now = datetime.now()
        
        # 策略1: 盈利策略
        for i in range(10):
            trade = Trade(
                trade_id=f"trade_1_{i}",
                symbol="EURUSD",
                type=TradeType.BUY,
                volume=0.1,
                open_price=1.1000 + i * 0.0001,
                close_price=1.1010 + i * 0.0001,
                sl=1.0990,
                tp=1.1020,
                profit=10.0 if i % 3 != 0 else -5.0,
                commission=0.5,
                swap=0.1,
                open_time=now - timedelta(days=10-i),
                close_time=now - timedelta(days=10-i, hours=-2),
                strategy_id="strategy_1"
            )
            self.performance_tracker.record_trade(trade)
        
        # 策略2: 亏损策略
        for i in range(5):
            trade = Trade(
                trade_id=f"trade_2_{i}",
                symbol="XAUUSD",
                type=TradeType.SELL,
                volume=0.01,
                open_price=1800.0 + i,
                close_price=1795.0 + i,
                sl=1810.0,
                tp=1790.0,
                profit=-8.0,
                commission=0.3,
                swap=-0.2,
                open_time=now - timedelta(days=5-i),
                close_time=now - timedelta(days=5-i, hours=-1),
                strategy_id="strategy_2"
            )
            self.performance_tracker.record_trade(trade)
        
        # 更新风险管理器的权益历史
        for i in range(30):
            equity = 10000 + i * 10 - (i % 5) * 20
            self.risk_manager.drawdown_monitor.update_equity(
                now - timedelta(days=30-i),
                equity
            )
    
    def _create_test_account(self) -> Account:
        """创建测试账户"""
        return Account(
            account_id="test_account",
            balance=10000.0,
            equity=10500.0,
            margin=500.0,
            free_margin=10000.0,
            margin_level=2100.0,
            currency="USD",
            leverage=100
        )
    
    def _create_test_positions(self) -> list:
        """创建测试持仓"""
        return [
            Position(
                position_id="pos_1",
                symbol="EURUSD",
                type=PositionType.LONG,
                volume=0.1,
                open_price=1.1000,
                current_price=1.1010,
                sl=1.0990,
                tp=1.1020,
                profit=10.0,
                swap=0.1,
                commission=0.5,
                open_time=datetime.now() - timedelta(hours=2)
            ),
            Position(
                position_id="pos_2",
                symbol="XAUUSD",
                type=PositionType.SHORT,
                volume=0.01,
                open_price=1800.0,
                current_price=1795.0,
                sl=1810.0,
                tp=1790.0,
                profit=5.0,
                swap=-0.2,
                commission=0.3,
                open_time=datetime.now() - timedelta(hours=1)
            )
        ]
    
    def test_generate_daily_report(self):
        """测试生成每日报告"""
        account = self._create_test_account()
        
        # 使用过去的日期，因为测试数据在过去10天
        report_date = datetime.now() - timedelta(days=5)
        
        report = self.report_generator.generate_daily_report(
            date=report_date,
            account=account
        )
        
        # 验证报告基本信息
        self.assertIsInstance(report, TradingReport)
        self.assertEqual(report.report_type, ReportType.DAILY)
        self.assertIsNotNone(report.report_id)
        self.assertTrue(report.report_id.startswith("daily_"))
        
        # 验证账户摘要
        self.assertIn('account_id', report.account_summary)
        self.assertEqual(report.account_summary['balance'], 10000.0)
        self.assertEqual(report.account_summary['equity'], 10500.0)
        
        # 验证交易统计（可能为0，因为日期范围可能没有交易）
        self.assertIn('total_trades', report.trading_summary)
        self.assertGreaterEqual(report.trading_summary['total_trades'], 0)
        
        # 验证策略表现
        self.assertIsInstance(report.strategy_performance, list)
        
        # 验证风险指标
        self.assertIn('max_drawdown', report.risk_metrics)
        
        # 验证合规检查
        self.assertIsInstance(report.compliance_results, list)
        
        # 验证报告已保存
        report_file = Path(self.temp_dir) / f"{report.report_id}.json"
        self.assertTrue(report_file.exists())
    
    def test_generate_weekly_report(self):
        """测试生成周报告"""
        account = self._create_test_account()
        
        report = self.report_generator.generate_weekly_report(
            week_start=datetime.now() - timedelta(days=7),
            account=account
        )
        
        # 验证报告类型
        self.assertEqual(report.report_type, ReportType.WEEKLY)
        self.assertTrue(report.report_id.startswith("weekly_"))
        
        # 验证周期
        period_duration = report.period_end - report.period_start
        self.assertEqual(period_duration.days, 7)
        
        # 验证报告内容
        self.assertIsNotNone(report.trading_summary)
        self.assertIsNotNone(report.risk_metrics)
    
    def test_generate_monthly_report(self):
        """测试生成月报告"""
        account = self._create_test_account()
        
        report = self.report_generator.generate_monthly_report(
            month_start=datetime.now().replace(day=1),
            account=account
        )
        
        # 验证报告类型
        self.assertEqual(report.report_type, ReportType.MONTHLY)
        self.assertTrue(report.report_id.startswith("monthly_"))
        
        # 验证报告内容
        self.assertIsNotNone(report.account_summary)
        self.assertIsNotNone(report.trading_summary)
        self.assertIsNotNone(report.strategy_performance)
    
    def test_generate_strategy_analysis_report(self):
        """测试生成策略分析报告"""
        report = self.report_generator.generate_strategy_analysis_report(
            strategy_names=["strategy_1", "strategy_2"],
            period=PerformancePeriod.MONTHLY
        )
        
        # 验证报告类型
        self.assertEqual(report.report_type, ReportType.STRATEGY_ANALYSIS)
        
        # 验证策略表现
        self.assertEqual(len(report.strategy_performance), 2)
        
        # 验证策略对比
        self.assertIn('strategies_compared', report.trading_summary)
        self.assertEqual(report.trading_summary['strategies_compared'], 2)
        
        # 验证策略排名
        self.assertIn('rankings', report.trading_summary)
        rankings = report.trading_summary['rankings']
        self.assertIn('by_net_profit', rankings)
        self.assertIn('by_win_rate', rankings)
        
        # 验证建议
        self.assertIsInstance(report.recommendations, list)
        self.assertGreater(len(report.recommendations), 0)
    
    def test_generate_risk_analysis_report(self):
        """测试生成风险分析报告"""
        account = self._create_test_account()
        positions = self._create_test_positions()
        
        report = self.report_generator.generate_risk_analysis_report(
            account=account,
            positions=positions,
            period=PerformancePeriod.MONTHLY
        )
        
        # 验证报告类型
        self.assertEqual(report.report_type, ReportType.RISK_ANALYSIS)
        
        # 验证风险指标
        self.assertIn('total_exposure', report.risk_metrics)
        self.assertIn('exposure_pct', report.risk_metrics)
        self.assertIn('position_count', report.risk_metrics)
        self.assertIn('margin_usage_pct', report.risk_metrics)
        
        # 验证持仓分析
        self.assertIn('total_positions', report.trading_summary)
        self.assertEqual(report.trading_summary['total_positions'], 2)
        self.assertIn('positions_by_symbol', report.trading_summary)
        
        # 验证建议和警告
        self.assertIsInstance(report.recommendations, list)
        self.assertIsInstance(report.warnings, list)
    
    def test_generate_compliance_report(self):
        """测试生成合规检查报告"""
        account = self._create_test_account()
        
        report = self.report_generator.generate_compliance_report(
            account=account,
            period=PerformancePeriod.MONTHLY
        )
        
        # 验证报告类型
        self.assertEqual(report.report_type, ReportType.COMPLIANCE)
        
        # 验证合规检查结果
        self.assertIsInstance(report.compliance_results, list)
        self.assertGreater(len(report.compliance_results), 0)
        
        # 验证每个合规检查结果
        for result in report.compliance_results:
            self.assertIsInstance(result, ComplianceCheckResult)
            self.assertIsNotNone(result.rule_id)
            self.assertIsNotNone(result.rule_name)
            self.assertIsInstance(result.passed, bool)
            self.assertIsNotNone(result.message)
        
        # 验证合规摘要
        self.assertIn('compliance_summary', report.trading_summary)
        summary = report.trading_summary['compliance_summary']
        self.assertIn('total_checks', summary)
        self.assertIn('passed_checks', summary)
        self.assertIn('compliance_rate', summary)
    
    def test_compliance_rules(self):
        """测试合规规则"""
        # 测试单笔最大风险规则
        rule = MaxRiskPerTradeRule("test_rule", "测试规则", 0.02)
        
        # 通过的情况
        passed, message = rule.check(0.01)
        self.assertTrue(passed)
        
        # 不通过的情况
        passed, message = rule.check(0.03)
        self.assertFalse(passed)
        self.assertIn("超过限制", message)
        
        # 测试日最大回撤规则
        dd_rule = MaxDailyDrawdownRule("dd_rule", "回撤规则", 0.05)
        
        passed, message = dd_rule.check(0.03)
        self.assertTrue(passed)
        
        passed, message = dd_rule.check(0.08)
        self.assertFalse(passed)
    
    def test_report_to_dict(self):
        """测试报告转换为字典"""
        account = self._create_test_account()
        report = self.report_generator.generate_daily_report(account=account)
        
        report_dict = report.to_dict()
        
        # 验证字典结构
        self.assertIsInstance(report_dict, dict)
        self.assertIn('report_id', report_dict)
        self.assertIn('report_type', report_dict)
        self.assertIn('period_start', report_dict)
        self.assertIn('period_end', report_dict)
        self.assertIn('account_summary', report_dict)
        self.assertIn('trading_summary', report_dict)
        self.assertIn('strategy_performance', report_dict)
        self.assertIn('risk_metrics', report_dict)
        self.assertIn('compliance_results', report_dict)
        self.assertIn('recommendations', report_dict)
        self.assertIn('warnings', report_dict)
    
    def test_report_to_json(self):
        """测试报告转换为JSON"""
        account = self._create_test_account()
        report = self.report_generator.generate_daily_report(account=account)
        
        json_str = report.to_json()
        
        # 验证JSON格式
        self.assertIsInstance(json_str, str)
        
        # 验证可以解析
        parsed = json.loads(json_str)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed['report_id'], report.report_id)
    
    def test_save_report_formats(self):
        """测试保存不同格式的报告"""
        account = self._create_test_account()
        report = self.report_generator.generate_daily_report(account=account)
        
        # 测试JSON格式（默认已保存）
        json_file = Path(self.temp_dir) / f"{report.report_id}.json"
        self.assertTrue(json_file.exists())
        
        # 测试Markdown格式
        self.report_generator._save_report(report, ReportFormat.MARKDOWN)
        md_file = Path(self.temp_dir) / f"{report.report_id}.markdown"
        self.assertTrue(md_file.exists())
        
        # 测试文本格式
        self.report_generator._save_report(report, ReportFormat.TEXT)
        txt_file = Path(self.temp_dir) / f"{report.report_id}.text"
        self.assertTrue(txt_file.exists())
    
    def test_recommendations_generation(self):
        """测试建议生成"""
        account = self._create_test_account()
        report = self.report_generator.generate_daily_report(account=account)
        
        # 验证有建议或警告
        self.assertTrue(
            len(report.recommendations) > 0 or len(report.warnings) > 0
        )
        
        # 验证建议格式
        for rec in report.recommendations:
            self.assertIsInstance(rec, str)
            self.assertGreater(len(rec), 0)
        
        # 验证警告格式
        for warn in report.warnings:
            self.assertIsInstance(warn, str)
            self.assertGreater(len(warn), 0)
    
    def test_strategy_rankings(self):
        """测试策略排名"""
        report = self.report_generator.generate_strategy_analysis_report(
            strategy_names=["strategy_1", "strategy_2"]
        )
        
        rankings = report.trading_summary['rankings']
        
        # 验证排名类型
        self.assertIn('by_net_profit', rankings)
        self.assertIn('by_win_rate', rankings)
        self.assertIn('by_sharpe_ratio', rankings)
        self.assertIn('by_profit_factor', rankings)
        
        # 验证排名数据
        for ranking_type, ranking_list in rankings.items():
            self.assertIsInstance(ranking_list, list)
            for item in ranking_list:
                self.assertIn('strategy', item)
                self.assertIn('value', item)
    
    def test_position_analysis(self):
        """测试持仓分析"""
        account = self._create_test_account()
        positions = self._create_test_positions()
        
        report = self.report_generator.generate_risk_analysis_report(
            account=account,
            positions=positions
        )
        
        analysis = report.trading_summary
        
        # 验证持仓统计
        self.assertEqual(analysis['total_positions'], 2)
        self.assertIn('long_positions', analysis)
        self.assertIn('short_positions', analysis)
        
        # 验证按品种统计
        self.assertIn('positions_by_symbol', analysis)
        self.assertIn('EURUSD', analysis['positions_by_symbol'])
        self.assertIn('XAUUSD', analysis['positions_by_symbol'])
        
        # 验证最大最小持仓
        self.assertIn('largest_position', analysis)
        self.assertIn('smallest_position', analysis)
    
    def test_empty_data_handling(self):
        """测试空数据处理"""
        # 创建空的性能跟踪器
        empty_tracker = PerformanceTracker()
        empty_generator = ReportGenerator(
            performance_tracker=empty_tracker,
            output_dir=self.temp_dir
        )
        
        # 生成报告不应该崩溃
        report = empty_generator.generate_daily_report()
        
        self.assertIsNotNone(report)
        self.assertEqual(report.trading_summary.get('total_trades', 0), 0)
    
    def test_period_calculation(self):
        """测试周期计算"""
        end_date = datetime(2024, 1, 15, 12, 0, 0)
        
        # 测试日周期
        start_date = self.report_generator._get_period_start_date(
            PerformancePeriod.DAILY, end_date
        )
        self.assertEqual((end_date - start_date).days, 1)
        
        # 测试周周期
        start_date = self.report_generator._get_period_start_date(
            PerformancePeriod.WEEKLY, end_date
        )
        self.assertEqual((end_date - start_date).days, 7)
        
        # 测试月周期
        start_date = self.report_generator._get_period_start_date(
            PerformancePeriod.MONTHLY, end_date
        )
        self.assertEqual((end_date - start_date).days, 30)


class TestComplianceRules(unittest.TestCase):
    """合规规则测试类"""
    
    def test_max_risk_per_trade_rule(self):
        """测试单笔最大风险规则"""
        rule = MaxRiskPerTradeRule("max_risk", "最大风险", 0.02)
        
        # 测试通过
        passed, msg = rule.check(0.015)
        self.assertTrue(passed)
        self.assertIn("符合要求", msg)
        
        # 测试不通过
        passed, msg = rule.check(0.025)
        self.assertFalse(passed)
        self.assertIn("超过限制", msg)
        
        # 测试边界值
        passed, msg = rule.check(0.02)
        self.assertTrue(passed)
    
    def test_max_drawdown_rule(self):
        """测试最大回撤规则"""
        rule = MaxDailyDrawdownRule("max_dd", "最大回撤", 0.05)
        
        passed, msg = rule.check(0.03)
        self.assertTrue(passed)
        
        passed, msg = rule.check(0.08)
        self.assertFalse(passed)


if __name__ == '__main__':
    unittest.main()
