"""
自定义技术指标模块
实现多周期分析和高级技术指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from src.core.models import MarketData
from src.core.exceptions import IndicatorCalculationError
from src.strategies.indicators import IndicatorResult, TechnicalIndicators


class MultiTimeframeAnalyzer:
    """多周期分析器"""
    
    def __init__(self, indicators: TechnicalIndicators):
        self.indicators = indicators
        self.logger = logging.getLogger(__name__)
        
        # 周期权重配置
        self.timeframe_weights = {
            "M1": 0.1,
            "M5": 0.15,
            "M15": 0.2,
            "M30": 0.25,
            "H1": 0.3,
            "H4": 0.4,
            "D1": 0.5
        }
    
    def analyze_trend_confluence(self, market_data_dict: Dict[str, MarketData]) -> Dict[str, Any]:
        """分析多周期趋势一致性"""
        trend_signals = {}
        
        for timeframe, market_data in market_data_dict.items():
            try:
                # 计算多个趋势指标
                sma_20 = self.indicators.calculate_sma(market_data, 20)
                ema_20 = self.indicators.calculate_ema(market_data, 20)
                macd = self.indicators.calculate_macd(market_data)
                adx = self.indicators.calculate_adx(market_data)
                
                current_price = market_data.ohlcv['close'].iloc[-1]
                
                # 趋势评分
                trend_score = 0
                signals = []
                
                # SMA趋势
                sma_value = sma_20.get_latest_value()
                if sma_value and current_price > sma_value:
                    trend_score += 1
                    signals.append("SMA_BULLISH")
                elif sma_value and current_price < sma_value:
                    trend_score -= 1
                    signals.append("SMA_BEARISH")
                
                # EMA趋势
                ema_value = ema_20.get_latest_value()
                if ema_value and current_price > ema_value:
                    trend_score += 1
                    signals.append("EMA_BULLISH")
                elif ema_value and current_price < ema_value:
                    trend_score -= 1
                    signals.append("EMA_BEARISH")
                
                # MACD趋势
                macd_value = macd.get_latest_value("macd")
                macd_signal = macd.get_latest_value("signal")
                if macd_value and macd_signal:
                    if macd_value > macd_signal:
                        trend_score += 1
                        signals.append("MACD_BULLISH")
                    else:
                        trend_score -= 1
                        signals.append("MACD_BEARISH")
                
                # ADX强度
                adx_value = adx.get_latest_value("adx")
                trend_strength = "weak"
                if adx_value:
                    if adx_value > 25:
                        trend_strength = "strong"
                    elif adx_value > 15:
                        trend_strength = "medium"
                
                trend_signals[timeframe] = {
                    "score": trend_score,
                    "strength": trend_strength,
                    "signals": signals,
                    "weight": self.timeframe_weights.get(timeframe, 0.1)
                }
                
            except Exception as e:
                self.logger.error(f"分析{timeframe}周期趋势失败: {str(e)}")
                trend_signals[timeframe] = {"error": str(e)}
        
        # 计算加权趋势评分
        weighted_score = 0
        total_weight = 0
        
        for timeframe, data in trend_signals.items():
            if "score" in data:
                weight = data["weight"]
                weighted_score += data["score"] * weight
                total_weight += weight
        
        final_score = weighted_score / total_weight if total_weight > 0 else 0
        
        return {
            "timeframe_analysis": trend_signals,
            "weighted_trend_score": final_score,
            "overall_trend": self._interpret_trend_score(final_score),
            "confluence_strength": self._calculate_confluence_strength(trend_signals)
        }
    
    def analyze_support_resistance(self, market_data_dict: Dict[str, MarketData]) -> Dict[str, Any]:
        """分析多周期支撑阻力位"""
        sr_levels = {}
        
        for timeframe, market_data in market_data_dict.items():
            try:
                ohlcv = market_data.ohlcv
                
                # 计算支撑阻力位
                support_levels = self._find_support_levels(ohlcv)
                resistance_levels = self._find_resistance_levels(ohlcv)
                
                # 计算关键价位
                current_price = ohlcv['close'].iloc[-1]
                nearest_support = self._find_nearest_level(current_price, support_levels, "below")
                nearest_resistance = self._find_nearest_level(current_price, resistance_levels, "above")
                
                sr_levels[timeframe] = {
                    "support_levels": support_levels,
                    "resistance_levels": resistance_levels,
                    "nearest_support": nearest_support,
                    "nearest_resistance": nearest_resistance,
                    "current_price": current_price,
                    "weight": self.timeframe_weights.get(timeframe, 0.1)
                }
                
            except Exception as e:
                self.logger.error(f"分析{timeframe}周期支撑阻力失败: {str(e)}")
                sr_levels[timeframe] = {"error": str(e)}
        
        # 合并多周期支撑阻力位
        consolidated_sr = self._consolidate_sr_levels(sr_levels)
        
        return {
            "timeframe_sr": sr_levels,
            "consolidated_support": consolidated_sr["support"],
            "consolidated_resistance": consolidated_sr["resistance"],
            "key_levels": consolidated_sr["key_levels"]
        }
    
    def _find_support_levels(self, ohlcv: pd.DataFrame, window: int = 10) -> List[float]:
        """寻找支撑位"""
        lows = ohlcv['low'].values
        support_levels = []
        
        for i in range(window, len(lows) - window):
            if lows[i] == min(lows[i-window:i+window+1]):
                support_levels.append(lows[i])
        
        # 去重并排序
        support_levels = sorted(list(set(support_levels)))
        
        # 合并相近的支撑位
        merged_levels = []
        for level in support_levels:
            if not merged_levels or abs(level - merged_levels[-1]) / merged_levels[-1] > 0.001:
                merged_levels.append(level)
        
        return merged_levels[-10:]  # 返回最近的10个支撑位
    
    def _find_resistance_levels(self, ohlcv: pd.DataFrame, window: int = 10) -> List[float]:
        """寻找阻力位"""
        highs = ohlcv['high'].values
        resistance_levels = []
        
        for i in range(window, len(highs) - window):
            if highs[i] == max(highs[i-window:i+window+1]):
                resistance_levels.append(highs[i])
        
        # 去重并排序
        resistance_levels = sorted(list(set(resistance_levels)), reverse=True)
        
        # 合并相近的阻力位
        merged_levels = []
        for level in resistance_levels:
            if not merged_levels or abs(level - merged_levels[-1]) / merged_levels[-1] > 0.001:
                merged_levels.append(level)
        
        return merged_levels[-10:]  # 返回最近的10个阻力位
    
    def _find_nearest_level(self, current_price: float, levels: List[float], 
                           direction: str) -> Optional[float]:
        """寻找最近的支撑或阻力位"""
        if not levels:
            return None
        
        if direction == "below":
            # 寻找最近的下方支撑位
            below_levels = [level for level in levels if level < current_price]
            return max(below_levels) if below_levels else None
        else:
            # 寻找最近的上方阻力位
            above_levels = [level for level in levels if level > current_price]
            return min(above_levels) if above_levels else None
    
    def _consolidate_sr_levels(self, sr_levels: Dict[str, Any]) -> Dict[str, Any]:
        """合并多周期支撑阻力位"""
        all_support = []
        all_resistance = []
        
        for timeframe, data in sr_levels.items():
            if "support_levels" in data:
                weight = data["weight"]
                for level in data["support_levels"]:
                    all_support.append({"level": level, "weight": weight, "timeframe": timeframe})
            
            if "resistance_levels" in data:
                weight = data["weight"]
                for level in data["resistance_levels"]:
                    all_resistance.append({"level": level, "weight": weight, "timeframe": timeframe})
        
        # 按权重排序并去重
        consolidated_support = self._merge_similar_levels(all_support)
        consolidated_resistance = self._merge_similar_levels(all_resistance)
        
        # 识别关键价位
        key_levels = self._identify_key_levels(consolidated_support, consolidated_resistance)
        
        return {
            "support": consolidated_support,
            "resistance": consolidated_resistance,
            "key_levels": key_levels
        }
    
    def _merge_similar_levels(self, levels: List[Dict[str, Any]], 
                             threshold: float = 0.001) -> List[Dict[str, Any]]:
        """合并相似的价位"""
        if not levels:
            return []
        
        # 按价位排序
        levels.sort(key=lambda x: x["level"])
        
        merged = []
        current_group = [levels[0]]
        
        for level in levels[1:]:
            # 检查是否与当前组相似
            avg_level = sum(l["level"] for l in current_group) / len(current_group)
            if abs(level["level"] - avg_level) / avg_level <= threshold:
                current_group.append(level)
            else:
                # 合并当前组
                merged_level = self._merge_level_group(current_group)
                merged.append(merged_level)
                current_group = [level]
        
        # 处理最后一组
        if current_group:
            merged_level = self._merge_level_group(current_group)
            merged.append(merged_level)
        
        return merged
    
    def _merge_level_group(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """合并一组相似的价位"""
        total_weight = sum(item["weight"] for item in group)
        weighted_level = sum(item["level"] * item["weight"] for item in group) / total_weight
        
        timeframes = list(set(item["timeframe"] for item in group))
        
        return {
            "level": weighted_level,
            "weight": total_weight,
            "timeframes": timeframes,
            "strength": len(timeframes)  # 出现在多少个周期中
        }
    
    def _identify_key_levels(self, support: List[Dict[str, Any]], 
                           resistance: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """识别关键价位"""
        all_levels = support + resistance
        
        # 按强度和权重排序
        key_levels = sorted(all_levels, 
                          key=lambda x: (x["strength"], x["weight"]), 
                          reverse=True)
        
        return key_levels[:5]  # 返回前5个关键价位
    
    def _interpret_trend_score(self, score: float) -> str:
        """解释趋势评分"""
        if score > 1.5:
            return "strong_bullish"
        elif score > 0.5:
            return "bullish"
        elif score > -0.5:
            return "neutral"
        elif score > -1.5:
            return "bearish"
        else:
            return "strong_bearish"
    
    def _calculate_confluence_strength(self, trend_signals: Dict[str, Any]) -> float:
        """计算趋势一致性强度"""
        scores = []
        weights = []
        
        for timeframe, data in trend_signals.items():
            if "score" in data:
                scores.append(abs(data["score"]))
                weights.append(data["weight"])
        
        if not scores:
            return 0.0
        
        # 计算加权标准差（一致性越高，标准差越小）
        weighted_mean = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        weighted_variance = sum(w * (s - weighted_mean) ** 2 for s, w in zip(scores, weights)) / sum(weights)
        weighted_std = np.sqrt(weighted_variance)
        
        # 转换为一致性评分（0-1，1表示完全一致）
        max_possible_std = max(scores) if scores else 1
        confluence_strength = 1 - (weighted_std / max_possible_std)
        
        return max(0, min(1, confluence_strength))


class CustomIndicators:
    """自定义指标计算器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_market_structure(self, market_data: MarketData) -> IndicatorResult:
        """计算市场结构指标"""
        try:
            ohlcv = market_data.ohlcv
            
            # 计算高点和低点
            highs = ohlcv['high'].values
            lows = ohlcv['low'].values
            
            # 识别摆动高点和低点
            swing_highs = self._find_swing_points(highs, "high")
            swing_lows = self._find_swing_points(lows, "low")
            
            # 分析市场结构
            structure_analysis = self._analyze_market_structure(swing_highs, swing_lows)
            
            return IndicatorResult(
                name="Market Structure",
                symbol=market_data.symbol,
                timeframe=market_data.timeframe,
                timestamp=market_data.timestamp,
                values={
                    "swing_highs": swing_highs,
                    "swing_lows": swing_lows,
                    "structure": structure_analysis["structure"],
                    "trend": structure_analysis["trend"],
                    "strength": structure_analysis["strength"]
                },
                metadata=structure_analysis
            )
            
        except Exception as e:
            raise IndicatorCalculationError(f"市场结构计算失败: {str(e)}")
    
    def calculate_volume_profile(self, market_data: MarketData, 
                               bins: int = 20) -> IndicatorResult:
        """计算成交量分布"""
        try:
            ohlcv = market_data.ohlcv
            
            if 'volume' not in ohlcv.columns:
                raise IndicatorCalculationError("缺少成交量数据")
            
            # 价格范围
            price_min = ohlcv['low'].min()
            price_max = ohlcv['high'].max()
            price_bins = np.linspace(price_min, price_max, bins + 1)
            
            # 计算每个价格区间的成交量
            volume_profile = np.zeros(bins)
            
            for _, row in ohlcv.iterrows():
                # 假设成交量在OHLC范围内均匀分布
                price_range = np.linspace(row['low'], row['high'], 10)
                volume_per_price = row['volume'] / len(price_range)
                
                for price in price_range:
                    bin_index = np.digitize(price, price_bins) - 1
                    if 0 <= bin_index < bins:
                        volume_profile[bin_index] += volume_per_price
            
            # 找到成交量最大的价格区间（POC - Point of Control）
            poc_index = np.argmax(volume_profile)
            poc_price = (price_bins[poc_index] + price_bins[poc_index + 1]) / 2
            
            # 计算价值区域（Value Area）
            value_area = self._calculate_value_area(volume_profile, price_bins)
            
            return IndicatorResult(
                name="Volume Profile",
                symbol=market_data.symbol,
                timeframe=market_data.timeframe,
                timestamp=market_data.timestamp,
                values={
                    "volume_profile": volume_profile,
                    "price_bins": price_bins,
                    "poc_price": poc_price,
                    "value_area_high": value_area["high"],
                    "value_area_low": value_area["low"]
                },
                metadata={"bins": bins, "total_volume": ohlcv['volume'].sum()}
            )
            
        except Exception as e:
            raise IndicatorCalculationError(f"成交量分布计算失败: {str(e)}")
    
    def calculate_order_flow(self, market_data: MarketData) -> IndicatorResult:
        """计算订单流指标"""
        try:
            ohlcv = market_data.ohlcv
            
            # 计算买卖压力
            buying_pressure = []
            selling_pressure = []
            
            for i in range(1, len(ohlcv)):
                prev_close = ohlcv['close'].iloc[i-1]
                current = ohlcv.iloc[i]
                
                # 简化的买卖压力计算
                if current['close'] > prev_close:
                    # 上涨，更多买压
                    buy_vol = current['volume'] * (current['close'] - current['low']) / (current['high'] - current['low'])
                    sell_vol = current['volume'] - buy_vol
                else:
                    # 下跌，更多卖压
                    sell_vol = current['volume'] * (current['high'] - current['close']) / (current['high'] - current['low'])
                    buy_vol = current['volume'] - sell_vol
                
                buying_pressure.append(buy_vol)
                selling_pressure.append(sell_vol)
            
            # 计算累积订单流
            cumulative_delta = np.cumsum(np.array(buying_pressure) - np.array(selling_pressure))
            
            # 计算订单流强度
            flow_strength = self._calculate_flow_strength(buying_pressure, selling_pressure)
            
            return IndicatorResult(
                name="Order Flow",
                symbol=market_data.symbol,
                timeframe=market_data.timeframe,
                timestamp=market_data.timestamp,
                values={
                    "buying_pressure": np.array(buying_pressure),
                    "selling_pressure": np.array(selling_pressure),
                    "cumulative_delta": cumulative_delta,
                    "flow_strength": flow_strength
                },
                metadata={"calculation_method": "simplified_hlc"}
            )
            
        except Exception as e:
            raise IndicatorCalculationError(f"订单流计算失败: {str(e)}")
    
    def _find_swing_points(self, data: np.ndarray, point_type: str, 
                          window: int = 5) -> List[Tuple[int, float]]:
        """寻找摆动高点或低点"""
        swing_points = []
        
        for i in range(window, len(data) - window):
            if point_type == "high":
                if data[i] == max(data[i-window:i+window+1]):
                    swing_points.append((i, data[i]))
            else:  # low
                if data[i] == min(data[i-window:i+window+1]):
                    swing_points.append((i, data[i]))
        
        return swing_points
    
    def _analyze_market_structure(self, swing_highs: List[Tuple[int, float]], 
                                swing_lows: List[Tuple[int, float]]) -> Dict[str, Any]:
        """分析市场结构"""
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {
                "structure": "insufficient_data",
                "trend": "unknown",
                "strength": 0
            }
        
        # 分析高点趋势
        recent_highs = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs
        high_trend = "neutral"
        if len(recent_highs) >= 2:
            if all(recent_highs[i][1] > recent_highs[i-1][1] for i in range(1, len(recent_highs))):
                high_trend = "higher_highs"
            elif all(recent_highs[i][1] < recent_highs[i-1][1] for i in range(1, len(recent_highs))):
                high_trend = "lower_highs"
        
        # 分析低点趋势
        recent_lows = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows
        low_trend = "neutral"
        if len(recent_lows) >= 2:
            if all(recent_lows[i][1] > recent_lows[i-1][1] for i in range(1, len(recent_lows))):
                low_trend = "higher_lows"
            elif all(recent_lows[i][1] < recent_lows[i-1][1] for i in range(1, len(recent_lows))):
                low_trend = "lower_lows"
        
        # 确定整体结构
        if high_trend == "higher_highs" and low_trend == "higher_lows":
            structure = "uptrend"
            trend = "bullish"
            strength = 0.8
        elif high_trend == "lower_highs" and low_trend == "lower_lows":
            structure = "downtrend"
            trend = "bearish"
            strength = 0.8
        elif high_trend == "higher_highs" and low_trend == "lower_lows":
            structure = "expanding"
            trend = "neutral"
            strength = 0.3
        elif high_trend == "lower_highs" and low_trend == "higher_lows":
            structure = "contracting"
            trend = "neutral"
            strength = 0.2
        else:
            structure = "sideways"
            trend = "neutral"
            strength = 0.1
        
        return {
            "structure": structure,
            "trend": trend,
            "strength": strength,
            "high_trend": high_trend,
            "low_trend": low_trend
        }
    
    def _calculate_value_area(self, volume_profile: np.ndarray, 
                            price_bins: np.ndarray, percentage: float = 0.7) -> Dict[str, float]:
        """计算价值区域"""
        total_volume = np.sum(volume_profile)
        target_volume = total_volume * percentage
        
        # 从POC开始向两边扩展
        poc_index = np.argmax(volume_profile)
        
        included_volume = volume_profile[poc_index]
        low_index = poc_index
        high_index = poc_index
        
        while included_volume < target_volume and (low_index > 0 or high_index < len(volume_profile) - 1):
            # 选择体积更大的一边扩展
            left_volume = volume_profile[low_index - 1] if low_index > 0 else 0
            right_volume = volume_profile[high_index + 1] if high_index < len(volume_profile) - 1 else 0
            
            if left_volume >= right_volume and low_index > 0:
                low_index -= 1
                included_volume += volume_profile[low_index]
            elif high_index < len(volume_profile) - 1:
                high_index += 1
                included_volume += volume_profile[high_index]
            else:
                break
        
        value_area_low = price_bins[low_index]
        value_area_high = price_bins[high_index + 1]
        
        return {
            "low": value_area_low,
            "high": value_area_high
        }
    
    def _calculate_flow_strength(self, buying_pressure: List[float], 
                               selling_pressure: List[float]) -> float:
        """计算订单流强度"""
        if not buying_pressure or not selling_pressure:
            return 0.0
        
        total_buying = sum(buying_pressure)
        total_selling = sum(selling_pressure)
        total_volume = total_buying + total_selling
        
        if total_volume == 0:
            return 0.0
        
        # 计算买卖比例
        buy_ratio = total_buying / total_volume
        
        # 转换为-1到1的强度值
        strength = (buy_ratio - 0.5) * 2
        
        return strength