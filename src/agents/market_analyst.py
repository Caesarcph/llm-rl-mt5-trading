"""
市场分析Agent
集成技术面和基本面分析，实现市场状态检测和相关性分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from src.core.models import MarketData, MarketState, Signal
from src.strategies.indicators import TechnicalIndicators, IndicatorResult
from src.core.exceptions import DataValidationError, AnalysisError


class TrendType(Enum):
    """趋势类型"""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"


class MarketRegime(Enum):
    """市场状态"""
    TRENDING = "trending"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    VOLATILE = "volatile"


class VolatilityLevel(Enum):
    """波动率水平"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class TrendAnalysis:
    """趋势分析结果"""
    trend_type: TrendType
    strength: float  # 0-1
    confidence: float  # 0-1
    duration_bars: int
    slope: float
    r_squared: float
    support_resistance: Dict[str, List[float]]
    
    def is_strong_trend(self, threshold: float = 0.7) -> bool:
        """判断是否为强趋势"""
        return self.strength >= threshold and self.confidence >= threshold


@dataclass
class VolatilityAnalysis:
    """波动率分析结果"""
    current_volatility: float
    volatility_level: VolatilityLevel
    volatility_percentile: float  # 历史百分位
    atr_ratio: float
    garch_forecast: Optional[float] = None
    
    def is_high_volatility(self, threshold: float = 0.75) -> bool:
        """判断是否为高波动率"""
        return self.volatility_percentile >= threshold


@dataclass
class CorrelationMatrix:
    """相关性矩阵"""
    symbols: List[str]
    correlation_matrix: pd.DataFrame
    timestamp: datetime
    lookback_period: int
    
    def get_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """获取两个品种的相关性"""
        if symbol1 in self.symbols and symbol2 in self.symbols:
            return self.correlation_matrix.loc[symbol1, symbol2]
        return None
    
    def get_highly_correlated_pairs(self, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """获取高相关性品种对"""
        pairs = []
        for i, symbol1 in enumerate(self.symbols):
            for j, symbol2 in enumerate(self.symbols[i+1:], i+1):
                corr = self.correlation_matrix.loc[symbol1, symbol2]
                if abs(corr) >= threshold:
                    pairs.append((symbol1, symbol2, corr))
        return pairs


@dataclass
class MarketAnalysis:
    """市场分析结果"""
    symbol: str
    timeframe: str
    timestamp: datetime
    trend_analysis: TrendAnalysis
    volatility_analysis: VolatilityAnalysis
    market_regime: MarketRegime
    technical_signals: Dict[str, float]
    support_resistance_levels: Dict[str, List[float]]
    market_score: float  # 综合市场评分 0-100
    
    def get_trading_recommendation(self) -> str:
        """获取交易建议"""
        if self.market_score >= 80:
            return "strong_buy" if self.trend_analysis.trend_type == TrendType.UPTREND else "strong_sell"
        elif self.market_score >= 60:
            return "buy" if self.trend_analysis.trend_type == TrendType.UPTREND else "sell"
        elif self.market_score <= 20:
            return "strong_sell" if self.trend_analysis.trend_type == TrendType.DOWNTREND else "strong_buy"
        elif self.market_score <= 40:
            return "sell" if self.trend_analysis.trend_type == TrendType.DOWNTREND else "buy"
        else:
            return "hold"


class MarketRegimeDetector:
    """市场状态检测器"""
    
    def __init__(self, lookback_period: int = 50):
        self.lookback_period = lookback_period
        self.logger = logging.getLogger(__name__)
    
    def detect_regime(self, market_data: MarketData, indicators: Dict[str, IndicatorResult]) -> MarketRegime:
        """检测市场状态"""
        try:
            # 获取价格数据
            prices = market_data.ohlcv['close'].tail(self.lookback_period)
            
            # 计算价格变化的统计特征
            returns = prices.pct_change().dropna()
            volatility = returns.std()
            
            # 获取技术指标
            adx_value = self._get_indicator_value(indicators, 'adx', 'adx')
            bb_bandwidth = self._calculate_bb_bandwidth(market_data)
            
            # 趋势强度分析
            trend_strength = adx_value if adx_value else 0
            
            # 波动率分析
            volatility_percentile = self._calculate_volatility_percentile(returns, volatility)
            
            # 价格区间分析
            price_range_ratio = self._calculate_price_range_ratio(prices)
            
            # 状态判断逻辑
            if trend_strength > 25 and price_range_ratio > 0.8:
                return MarketRegime.TRENDING
            elif volatility_percentile > 0.8 or bb_bandwidth > 0.05:
                if trend_strength > 20:
                    return MarketRegime.BREAKOUT
                else:
                    return MarketRegime.VOLATILE
            else:
                return MarketRegime.RANGING
                
        except Exception as e:
            self.logger.error(f"市场状态检测失败: {str(e)}")
            return MarketRegime.RANGING
    
    def _get_indicator_value(self, indicators: Dict[str, IndicatorResult], 
                           indicator_name: str, value_key: str) -> Optional[float]:
        """获取指标值"""
        if indicator_name in indicators:
            return indicators[indicator_name].get_latest_value(value_key)
        return None
    
    def _calculate_bb_bandwidth(self, market_data: MarketData) -> float:
        """计算布林带带宽"""
        try:
            from src.strategies.indicators import TechnicalIndicators
            tech_indicators = TechnicalIndicators()
            bb_result = tech_indicators.calculate_bollinger_bands(market_data)
            
            upper = bb_result.get_latest_value("upper")
            lower = bb_result.get_latest_value("lower")
            middle = bb_result.get_latest_value("middle")
            
            if all(v is not None for v in [upper, lower, middle]):
                return (upper - lower) / middle
            return 0.0
        except Exception:
            return 0.0
    
    def _calculate_volatility_percentile(self, returns: pd.Series, current_vol: float) -> float:
        """计算波动率百分位"""
        try:
            rolling_vol = returns.rolling(window=20).std()
            percentile = stats.percentileofscore(rolling_vol.dropna(), current_vol) / 100
            return percentile
        except Exception:
            return 0.5
    
    def _calculate_price_range_ratio(self, prices: pd.Series) -> float:
        """计算价格区间比率"""
        try:
            price_high = prices.max()
            price_low = prices.min()
            current_price = prices.iloc[-1]
            
            if price_high == price_low:
                return 0.5
            
            return (current_price - price_low) / (price_high - price_low)
        except Exception:
            return 0.5


class CorrelationAnalyzer:
    """相关性分析器"""
    
    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period
        self.logger = logging.getLogger(__name__)
    
    def analyze_correlations(self, market_data_dict: Dict[str, MarketData]) -> CorrelationMatrix:
        """分析品种间相关性"""
        try:
            # 准备价格数据
            price_data = {}
            for symbol, market_data in market_data_dict.items():
                prices = market_data.ohlcv['close'].tail(self.lookback_period)
                returns = prices.pct_change().dropna()
                price_data[symbol] = returns
            
            # 创建DataFrame
            df = pd.DataFrame(price_data)
            df = df.dropna()
            
            # 计算相关性矩阵
            correlation_matrix = df.corr()
            
            return CorrelationMatrix(
                symbols=list(market_data_dict.keys()),
                correlation_matrix=correlation_matrix,
                timestamp=datetime.now(),
                lookback_period=self.lookback_period
            )
            
        except Exception as e:
            self.logger.error(f"相关性分析失败: {str(e)}")
            # 返回空的相关性矩阵
            symbols = list(market_data_dict.keys())
            empty_matrix = pd.DataFrame(
                np.eye(len(symbols)), 
                index=symbols, 
                columns=symbols
            )
            return CorrelationMatrix(
                symbols=symbols,
                correlation_matrix=empty_matrix,
                timestamp=datetime.now(),
                lookback_period=self.lookback_period
            )
    
    def detect_regime_changes(self, correlation_history: List[CorrelationMatrix]) -> Dict[str, Any]:
        """检测相关性状态变化"""
        if len(correlation_history) < 2:
            return {"status": "insufficient_data"}
        
        try:
            current_corr = correlation_history[-1]
            previous_corr = correlation_history[-2]
            
            # 计算相关性变化
            corr_changes = {}
            significant_changes = []
            
            for symbol1 in current_corr.symbols:
                for symbol2 in current_corr.symbols:
                    if symbol1 != symbol2:
                        current_val = current_corr.get_correlation(symbol1, symbol2)
                        previous_val = previous_corr.get_correlation(symbol1, symbol2)
                        
                        if current_val is not None and previous_val is not None:
                            change = abs(current_val - previous_val)
                            corr_changes[f"{symbol1}_{symbol2}"] = change
                            
                            if change > 0.3:  # 显著变化阈值
                                significant_changes.append({
                                    "pair": (symbol1, symbol2),
                                    "change": change,
                                    "current": current_val,
                                    "previous": previous_val
                                })
            
            return {
                "status": "analyzed",
                "average_change": np.mean(list(corr_changes.values())),
                "max_change": max(corr_changes.values()) if corr_changes else 0,
                "significant_changes": significant_changes,
                "regime_shift_detected": len(significant_changes) > 0
            }
            
        except Exception as e:
            self.logger.error(f"相关性状态变化检测失败: {str(e)}")
            return {"status": "error", "message": str(e)}


class MarketAnalystAgent:
    """市场分析Agent"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.technical_indicators = TechnicalIndicators(
            use_cache=self.config.get('use_cache', True),
            cache_ttl=self.config.get('cache_ttl', 300)
        )
        self.regime_detector = MarketRegimeDetector(
            lookback_period=self.config.get('regime_lookback', 50)
        )
        self.correlation_analyzer = CorrelationAnalyzer(
            lookback_period=self.config.get('correlation_lookback', 100)
        )
        
        # 历史数据存储
        self.correlation_history: List[CorrelationMatrix] = []
        self.analysis_history: Dict[str, List[MarketAnalysis]] = {}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'use_cache': True,
            'cache_ttl': 300,
            'regime_lookback': 50,
            'correlation_lookback': 100,
            'trend_min_bars': 10,
            'volatility_window': 20,
            'support_resistance_strength': 3
        }
    
    def analyze_market_state(self, symbol: str, market_data: MarketData) -> MarketState:
        """分析市场状态"""
        try:
            # 计算技术指标
            indicators = self._calculate_key_indicators(market_data)
            
            # 趋势分析
            trend_analysis = self._analyze_trend(market_data, indicators)
            
            # 波动率分析
            volatility_analysis = self._analyze_volatility(market_data, indicators)
            
            # 检测市场状态
            market_regime = self.regime_detector.detect_regime(market_data, indicators)
            
            # 支撑阻力位分析
            sr_levels = self._find_support_resistance_levels(market_data)
            
            # 创建市场状态对象
            market_state = MarketState(
                trend=trend_analysis.trend_type.value,
                volatility=volatility_analysis.current_volatility,
                regime=market_regime.value,
                support_resistance=sr_levels
            )
            
            return market_state
            
        except Exception as e:
            self.logger.error(f"市场状态分析失败 {symbol}: {str(e)}")
            # 返回默认状态
            return MarketState(
                trend=TrendType.SIDEWAYS.value,
                volatility=0.02,
                regime=MarketRegime.RANGING.value,
                support_resistance={"support": [], "resistance": []}
            )
    
    def generate_comprehensive_analysis(self, symbol: str, market_data: MarketData) -> MarketAnalysis:
        """生成综合市场分析"""
        try:
            # 计算技术指标
            indicators = self._calculate_key_indicators(market_data)
            
            # 各项分析
            trend_analysis = self._analyze_trend(market_data, indicators)
            volatility_analysis = self._analyze_volatility(market_data, indicators)
            market_regime = self.regime_detector.detect_regime(market_data, indicators)
            
            # 技术信号汇总
            technical_signals = self._generate_technical_signals(indicators)
            
            # 支撑阻力位
            sr_levels = self._find_support_resistance_levels(market_data)
            
            # 计算综合市场评分
            market_score = self._calculate_market_score(
                trend_analysis, volatility_analysis, technical_signals
            )
            
            # 创建分析结果
            analysis = MarketAnalysis(
                symbol=symbol,
                timeframe=market_data.timeframe,
                timestamp=market_data.timestamp,
                trend_analysis=trend_analysis,
                volatility_analysis=volatility_analysis,
                market_regime=market_regime,
                technical_signals=technical_signals,
                support_resistance_levels=sr_levels,
                market_score=market_score
            )
            
            # 保存到历史记录
            if symbol not in self.analysis_history:
                self.analysis_history[symbol] = []
            self.analysis_history[symbol].append(analysis)
            
            # 保持历史记录数量限制
            if len(self.analysis_history[symbol]) > 100:
                self.analysis_history[symbol] = self.analysis_history[symbol][-100:]
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"综合市场分析失败 {symbol}: {str(e)}")
            raise AnalysisError(f"市场分析失败: {str(e)}")
    
    def analyze_multi_symbol_correlations(self, market_data_dict: Dict[str, MarketData]) -> CorrelationMatrix:
        """分析多品种相关性"""
        try:
            correlation_matrix = self.correlation_analyzer.analyze_correlations(market_data_dict)
            
            # 保存到历史记录
            self.correlation_history.append(correlation_matrix)
            
            # 保持历史记录数量限制
            if len(self.correlation_history) > 50:
                self.correlation_history = self.correlation_history[-50:]
            
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"多品种相关性分析失败: {str(e)}")
            raise AnalysisError(f"相关性分析失败: {str(e)}")
    
    def detect_correlation_regime_changes(self) -> Dict[str, Any]:
        """检测相关性状态变化"""
        return self.correlation_analyzer.detect_regime_changes(self.correlation_history)
    
    def _calculate_key_indicators(self, market_data: MarketData) -> Dict[str, IndicatorResult]:
        """计算关键技术指标"""
        indicators = {}
        
        try:
            # 趋势指标
            indicators['sma_20'] = self.technical_indicators.calculate_sma(market_data, period=20)
            indicators['ema_20'] = self.technical_indicators.calculate_ema(market_data, period=20)
            indicators['macd'] = self.technical_indicators.calculate_macd(market_data)
            indicators['adx'] = self.technical_indicators.calculate_adx(market_data)
            
            # 动量指标
            indicators['rsi'] = self.technical_indicators.calculate_rsi(market_data)
            indicators['stochastic'] = self.technical_indicators.calculate_stochastic(market_data)
            indicators['cci'] = self.technical_indicators.calculate_cci(market_data)
            
            # 波动率指标
            indicators['bollinger_bands'] = self.technical_indicators.calculate_bollinger_bands(market_data)
            indicators['atr'] = self.technical_indicators.calculate_atr(market_data)
            
        except Exception as e:
            self.logger.error(f"技术指标计算失败: {str(e)}")
        
        return indicators
    
    def _analyze_trend(self, market_data: MarketData, indicators: Dict[str, IndicatorResult]) -> TrendAnalysis:
        """分析趋势"""
        try:
            prices = market_data.ohlcv['close'].tail(50)
            
            # 线性回归分析趋势
            x = np.arange(len(prices))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
            
            # 趋势方向判断
            if slope > 0 and r_value > 0.7:
                trend_type = TrendType.UPTREND
            elif slope < 0 and r_value < -0.7:
                trend_type = TrendType.DOWNTREND
            else:
                trend_type = TrendType.SIDEWAYS
            
            # 趋势强度（基于ADX和线性回归）
            adx_value = indicators.get('adx', {}).get_latest_value('adx') or 0
            trend_strength = min(1.0, (adx_value / 50) * abs(r_value))
            
            # 趋势置信度
            confidence = abs(r_value)
            
            # 支撑阻力位
            sr_levels = self._find_support_resistance_levels(market_data)
            
            return TrendAnalysis(
                trend_type=trend_type,
                strength=trend_strength,
                confidence=confidence,
                duration_bars=len(prices),
                slope=slope,
                r_squared=r_value**2,
                support_resistance=sr_levels
            )
            
        except Exception as e:
            self.logger.error(f"趋势分析失败: {str(e)}")
            return TrendAnalysis(
                trend_type=TrendType.SIDEWAYS,
                strength=0.0,
                confidence=0.0,
                duration_bars=0,
                slope=0.0,
                r_squared=0.0,
                support_resistance={"support": [], "resistance": []}
            )
    
    def _analyze_volatility(self, market_data: MarketData, indicators: Dict[str, IndicatorResult]) -> VolatilityAnalysis:
        """分析波动率"""
        try:
            prices = market_data.ohlcv['close'].tail(100)
            returns = prices.pct_change().dropna()
            
            # 当前波动率
            current_volatility = returns.tail(20).std()
            
            # 历史波动率百分位
            rolling_vol = returns.rolling(window=20).std()
            volatility_percentile = stats.percentileofscore(rolling_vol.dropna(), current_volatility) / 100
            
            # ATR比率
            atr_value = indicators.get('atr', {}).get_latest_value() or 0
            current_price = prices.iloc[-1]
            atr_ratio = atr_value / current_price if current_price > 0 else 0
            
            # 波动率水平分类
            if volatility_percentile >= 0.8:
                volatility_level = VolatilityLevel.HIGH
            elif volatility_percentile >= 0.6:
                volatility_level = VolatilityLevel.MEDIUM
            elif volatility_percentile >= 0.2:
                volatility_level = VolatilityLevel.LOW
            else:
                volatility_level = VolatilityLevel.EXTREME
            
            return VolatilityAnalysis(
                current_volatility=current_volatility,
                volatility_level=volatility_level,
                volatility_percentile=volatility_percentile,
                atr_ratio=atr_ratio
            )
            
        except Exception as e:
            self.logger.error(f"波动率分析失败: {str(e)}")
            return VolatilityAnalysis(
                current_volatility=0.02,
                volatility_level=VolatilityLevel.MEDIUM,
                volatility_percentile=0.5,
                atr_ratio=0.01
            )
    
    def _find_support_resistance_levels(self, market_data: MarketData) -> Dict[str, List[float]]:
        """寻找支撑阻力位"""
        try:
            ohlc = market_data.ohlcv.tail(100)
            highs = ohlc['high'].values
            lows = ohlc['low'].values
            
            # 寻找局部极值点
            support_levels = []
            resistance_levels = []
            
            # 简化的支撑阻力位算法
            window = 5
            for i in range(window, len(lows) - window):
                # 支撑位（局部最低点）
                if all(lows[i] <= lows[i-j] for j in range(1, window+1)) and \
                   all(lows[i] <= lows[i+j] for j in range(1, window+1)):
                    support_levels.append(lows[i])
                
                # 阻力位（局部最高点）
                if all(highs[i] >= highs[i-j] for j in range(1, window+1)) and \
                   all(highs[i] >= highs[i+j] for j in range(1, window+1)):
                    resistance_levels.append(highs[i])
            
            # 去重并排序
            support_levels = sorted(list(set(support_levels)))[-5:]  # 保留最近5个
            resistance_levels = sorted(list(set(resistance_levels)), reverse=True)[:5]  # 保留最近5个
            
            return {
                "support": support_levels,
                "resistance": resistance_levels
            }
            
        except Exception as e:
            self.logger.error(f"支撑阻力位分析失败: {str(e)}")
            return {"support": [], "resistance": []}
    
    def _generate_technical_signals(self, indicators: Dict[str, IndicatorResult]) -> Dict[str, float]:
        """生成技术信号"""
        signals = {}
        
        try:
            # RSI信号
            rsi_value = indicators.get('rsi', {}).get_latest_value()
            if rsi_value:
                if rsi_value > 70:
                    signals['rsi_signal'] = -1.0  # 超买
                elif rsi_value < 30:
                    signals['rsi_signal'] = 1.0   # 超卖
                else:
                    signals['rsi_signal'] = 0.0   # 中性
            
            # MACD信号
            macd_value = indicators.get('macd', {}).get_latest_value('macd')
            macd_signal = indicators.get('macd', {}).get_latest_value('signal')
            if macd_value and macd_signal:
                signals['macd_signal'] = 1.0 if macd_value > macd_signal else -1.0
            
            # 随机指标信号
            stoch_k = indicators.get('stochastic', {}).get_latest_value('k')
            stoch_d = indicators.get('stochastic', {}).get_latest_value('d')
            if stoch_k and stoch_d:
                if stoch_k > 80 and stoch_d > 80:
                    signals['stoch_signal'] = -1.0  # 超买
                elif stoch_k < 20 and stoch_d < 20:
                    signals['stoch_signal'] = 1.0   # 超卖
                else:
                    signals['stoch_signal'] = 0.0   # 中性
            
            # ADX趋势强度信号
            adx_value = indicators.get('adx', {}).get_latest_value('adx')
            if adx_value:
                if adx_value > 25:
                    signals['trend_strength'] = 1.0  # 强趋势
                elif adx_value > 15:
                    signals['trend_strength'] = 0.5  # 中等趋势
                else:
                    signals['trend_strength'] = 0.0  # 弱趋势
            
        except Exception as e:
            self.logger.error(f"技术信号生成失败: {str(e)}")
        
        return signals
    
    def _calculate_market_score(self, trend_analysis: TrendAnalysis, 
                              volatility_analysis: VolatilityAnalysis,
                              technical_signals: Dict[str, float]) -> float:
        """计算综合市场评分"""
        try:
            score = 50.0  # 基础分数
            
            # 趋势评分 (30%)
            trend_score = trend_analysis.strength * trend_analysis.confidence * 30
            if trend_analysis.trend_type == TrendType.UPTREND:
                score += trend_score
            elif trend_analysis.trend_type == TrendType.DOWNTREND:
                score -= trend_score
            
            # 技术信号评分 (40%)
            signal_sum = sum(technical_signals.values())
            signal_score = (signal_sum / len(technical_signals)) * 20 if technical_signals else 0
            score += signal_score
            
            # 波动率评分 (20%)
            if volatility_analysis.volatility_level == VolatilityLevel.MEDIUM:
                score += 10  # 适中波动率加分
            elif volatility_analysis.volatility_level == VolatilityLevel.HIGH:
                score -= 5   # 高波动率减分
            elif volatility_analysis.volatility_level == VolatilityLevel.EXTREME:
                score -= 15  # 极端波动率大幅减分
            
            # 趋势强度评分 (10%)
            trend_strength_signal = technical_signals.get('trend_strength', 0)
            score += trend_strength_signal * 10
            
            # 确保评分在0-100范围内
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            self.logger.error(f"市场评分计算失败: {str(e)}")
            return 50.0
    
    def get_analysis_summary(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取分析汇总"""
        if symbol not in self.analysis_history or not self.analysis_history[symbol]:
            return None
        
        latest_analysis = self.analysis_history[symbol][-1]
        
        return {
            "symbol": symbol,
            "timestamp": latest_analysis.timestamp,
            "trend": latest_analysis.trend_analysis.trend_type.value,
            "trend_strength": latest_analysis.trend_analysis.strength,
            "market_regime": latest_analysis.market_regime.value,
            "volatility_level": latest_analysis.volatility_analysis.volatility_level.value,
            "market_score": latest_analysis.market_score,
            "recommendation": latest_analysis.get_trading_recommendation(),
            "support_levels": latest_analysis.support_resistance_levels.get("support", []),
            "resistance_levels": latest_analysis.support_resistance_levels.get("resistance", [])
        }
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self.technical_indicators.clear_cache()
        self.analysis_history.clear()
        self.correlation_history.clear()