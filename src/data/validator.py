#!/usr/bin/env python3
"""
数据验证器
确保数据完整性和一致性
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
import numpy as np

from .models import MarketData, Tick, ValidationResult, TimeFrame
from ..core.logging import get_logger


class DataValidator:
    """数据验证器"""
    
    def __init__(self):
        self.logger = get_logger()
        
        # 验证阈值配置
        self.max_spread_pct = 0.01  # 最大点差百分比
        self.max_price_change_pct = 0.1  # 最大价格变化百分比
        self.min_volume = 1  # 最小成交量
        self.max_gap_minutes = 60  # 最大数据间隔(分钟)
    
    def validate_market_data(self, data: MarketData) -> ValidationResult:
        """
        验证市场数据完整性
        
        Args:
            data: 市场数据
            
        Returns:
            ValidationResult: 验证结果
        """
        result = ValidationResult(is_valid=True)
        
        try:
            # 基础验证
            self._validate_basic_data(data, result)
            
            # OHLCV数据验证
            if data.ohlcv is not None and not data.ohlcv.empty:
                self._validate_ohlcv_data(data.ohlcv, result)
                self._validate_price_consistency(data.ohlcv, result)
                self._validate_volume_data(data.ohlcv, result)
                self._validate_data_continuity(data.ohlcv, data.timeframe, result)
            
            # 点差验证
            if data.spread > 0:
                self._validate_spread(data, result)
            
            # 指标数据验证
            if data.indicators:
                self._validate_indicators(data.indicators, result)
            
        except Exception as e:
            result.add_error(f"数据验证异常: {e}")
            self.logger.error(f"数据验证异常: {e}")
        
        return result
    
    def validate_tick_data(self, tick: Tick) -> ValidationResult:
        """
        验证Tick数据
        
        Args:
            tick: Tick数据
            
        Returns:
            ValidationResult: 验证结果
        """
        result = ValidationResult(is_valid=True)
        
        try:
            # 基础价格验证
            if tick.bid <= 0 or tick.ask <= 0:
                result.add_error("买价或卖价不能为零或负数")
            
            if tick.ask <= tick.bid:
                result.add_error("卖价必须大于买价")
            
            # 点差验证
            spread_pct = tick.spread / tick.mid_price
            if spread_pct > self.max_spread_pct:
                result.add_warning(f"点差过大: {spread_pct:.4f} > {self.max_spread_pct}")
            
            # 成交量验证
            if tick.volume < 0:
                result.add_error("成交量不能为负数")
            
            # 时间戳验证
            if tick.timestamp > datetime.now() + timedelta(minutes=5):
                result.add_error("时间戳不能超过当前时间太多")
            
        except Exception as e:
            result.add_error(f"Tick数据验证异常: {e}")
            self.logger.error(f"Tick数据验证异常: {e}")
        
        return result
    
    def _validate_basic_data(self, data: MarketData, result: ValidationResult):
        """验证基础数据"""
        if not data.symbol:
            result.add_error("交易品种不能为空")
        
        if not isinstance(data.timeframe, TimeFrame):
            result.add_error("时间周期格式错误")
        
        if not isinstance(data.timestamp, datetime):
            result.add_error("时间戳格式错误")
    
    def _validate_ohlcv_data(self, ohlcv: pd.DataFrame, result: ValidationResult):
        """验证OHLCV数据"""
        if ohlcv.empty:
            result.add_error("OHLCV数据为空")
            return
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in ohlcv.columns]
        if missing_columns:
            result.add_error(f"缺少必要列: {missing_columns}")
            return
        
        # 检查数据类型
        for col in ['open', 'high', 'low', 'close']:
            if not pd.api.types.is_numeric_dtype(ohlcv[col]):
                result.add_error(f"列 {col} 必须是数值类型")
        
        # 检查空值
        null_counts = ohlcv[required_columns].isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                result.add_warning(f"列 {col} 有 {count} 个空值")
        
        # 检查负值
        for col in required_columns:
            if (ohlcv[col] < 0).any():
                result.add_error(f"列 {col} 包含负值")
    
    def _validate_price_consistency(self, ohlcv: pd.DataFrame, result: ValidationResult):
        """验证价格一致性"""
        if ohlcv.empty:
            return
        
        # 检查 high >= max(open, close) 和 low <= min(open, close)
        high_valid = (ohlcv['high'] >= ohlcv[['open', 'close']].max(axis=1)).all()
        low_valid = (ohlcv['low'] <= ohlcv[['open', 'close']].min(axis=1)).all()
        
        if not high_valid:
            result.add_error("最高价必须大于等于开盘价和收盘价")
        
        if not low_valid:
            result.add_error("最低价必须小于等于开盘价和收盘价")
        
        # 检查价格跳跃
        if len(ohlcv) > 1:
            price_changes = ohlcv['close'].pct_change().abs()
            large_changes = price_changes > self.max_price_change_pct
            
            if large_changes.any():
                count = large_changes.sum()
                result.add_warning(f"发现 {count} 个异常价格跳跃 (>{self.max_price_change_pct*100}%)")
    
    def _validate_volume_data(self, ohlcv: pd.DataFrame, result: ValidationResult):
        """验证成交量数据"""
        if 'volume' not in ohlcv.columns:
            return
        
        # 检查负成交量
        if (ohlcv['volume'] < 0).any():
            result.add_error("成交量不能为负数")
        
        # 检查异常低成交量
        low_volume_count = (ohlcv['volume'] < self.min_volume).sum()
        if low_volume_count > 0:
            result.add_warning(f"发现 {low_volume_count} 个低成交量数据点")
        
        # 检查成交量异常值
        if len(ohlcv) > 10:
            volume_mean = ohlcv['volume'].mean()
            volume_std = ohlcv['volume'].std()
            
            if volume_std > 0:
                outliers = np.abs(ohlcv['volume'] - volume_mean) > 3 * volume_std
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    result.add_warning(f"发现 {outlier_count} 个成交量异常值")
    
    def _validate_data_continuity(self, ohlcv: pd.DataFrame, timeframe: TimeFrame, result: ValidationResult):
        """验证数据连续性"""
        if len(ohlcv) < 2:
            return
        
        # 获取时间间隔(分钟)
        timeframe_minutes = self._get_timeframe_minutes(timeframe)
        if timeframe_minutes is None:
            return
        
        # 检查时间间隔
        if 'time' in ohlcv.columns:
            time_diffs = ohlcv['time'].diff().dt.total_seconds() / 60
            expected_diff = timeframe_minutes
            
            # 允许一定的误差
            tolerance = expected_diff * 0.1
            irregular_intervals = np.abs(time_diffs - expected_diff) > tolerance
            
            if irregular_intervals.any():
                count = irregular_intervals.sum()
                result.add_warning(f"发现 {count} 个不规则时间间隔")
        
        # 检查数据缺口
        gaps = self._detect_data_gaps(ohlcv, timeframe_minutes)
        if gaps:
            result.add_warning(f"发现 {len(gaps)} 个数据缺口")
    
    def _validate_spread(self, data: MarketData, result: ValidationResult):
        """验证点差数据"""
        if data.spread < 0:
            result.add_error("点差不能为负数")
        
        # 基于收盘价计算点差百分比
        if not data.ohlcv.empty and 'close' in data.ohlcv.columns:
            last_close = data.ohlcv['close'].iloc[-1]
            spread_pct = data.spread / last_close
            
            if spread_pct > self.max_spread_pct:
                result.add_warning(f"点差过大: {spread_pct:.4f} > {self.max_spread_pct}")
    
    def _validate_indicators(self, indicators: dict, result: ValidationResult):
        """验证技术指标数据"""
        for name, value in indicators.items():
            if not isinstance(value, (int, float)):
                result.add_error(f"指标 {name} 值必须是数值类型")
                continue
            
            if np.isnan(value) or np.isinf(value):
                result.add_warning(f"指标 {name} 值为 NaN 或无穷大")
    
    def _get_timeframe_minutes(self, timeframe: TimeFrame) -> Optional[int]:
        """获取时间周期对应的分钟数"""
        timeframe_map = {
            TimeFrame.M1: 1,
            TimeFrame.M5: 5,
            TimeFrame.M15: 15,
            TimeFrame.M30: 30,
            TimeFrame.H1: 60,
            TimeFrame.H4: 240,
            TimeFrame.D1: 1440,
            TimeFrame.W1: 10080,
            TimeFrame.MN1: 43200,  # 近似值
        }
        return timeframe_map.get(timeframe)
    
    def _detect_data_gaps(self, ohlcv: pd.DataFrame, expected_interval_minutes: int) -> List[tuple]:
        """检测数据缺口"""
        gaps = []
        
        if 'time' not in ohlcv.columns or len(ohlcv) < 2:
            return gaps
        
        for i in range(1, len(ohlcv)):
            time_diff = (ohlcv['time'].iloc[i] - ohlcv['time'].iloc[i-1]).total_seconds() / 60
            
            if time_diff > expected_interval_minutes * 1.5:  # 允许50%的误差
                gaps.append((ohlcv['time'].iloc[i-1], ohlcv['time'].iloc[i]))
        
        return gaps
    
    def get_data_quality_score(self, validation_result: ValidationResult) -> float:
        """
        计算数据质量评分
        
        Args:
            validation_result: 验证结果
            
        Returns:
            float: 质量评分 (0-1)
        """
        if not validation_result.is_valid:
            return 0.0
        
        # 基础分数
        score = 1.0
        
        # 每个警告扣除0.1分
        warning_penalty = len(validation_result.warnings) * 0.1
        score = max(0.0, score - warning_penalty)
        
        return score