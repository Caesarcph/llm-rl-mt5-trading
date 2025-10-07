"""
订单执行器模块
处理订单发送、确认、滑点预测和错误处理
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import MetaTrader5 as mt5

from src.core.models import Signal, Position, PositionType, OrderStatus
from src.core.exceptions import OrderExecutionError, ConnectionError


logger = logging.getLogger(__name__)


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class SlippageModel(Enum):
    """滑点模型"""
    FIXED = "fixed"  # 固定滑点
    DYNAMIC = "dynamic"  # 动态滑点
    HISTORICAL = "historical"  # 基于历史数据


class OrderExecutor:
    """订单执行器类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化订单执行器
        
        Args:
            config: 配置字典，包含滑点设置、重试次数等
        """
        self.config = config or {}
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)
        self.max_slippage = self.config.get('max_slippage', 0.0005)  # 最大滑点0.05%
        self.slippage_model = SlippageModel(self.config.get('slippage_model', 'dynamic'))
        
        # 订单状态跟踪
        self.pending_orders: Dict[str, Dict[str, Any]] = {}
        self.order_history: List[Dict[str, Any]] = []
        
        # 滑点历史数据
        self.slippage_history: Dict[str, List[float]] = {}
        
        logger.info(f"订单执行器初始化完成，最大重试次数: {self.max_retries}")
    
    def send_order(self, signal: Signal, account_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        发送订单
        
        Args:
            signal: 交易信号
            account_info: 账户信息（可选）
            
        Returns:
            订单结果字典
        """
        logger.info(f"准备发送订单: {signal.symbol} {signal.direction} {signal.size}手")
        
        # 验证信号
        if not self._validate_signal(signal):
            return self._create_order_result(
                success=False,
                error="信号验证失败",
                signal=signal
            )
        
        # 预测滑点
        predicted_slippage = self.predict_slippage(signal.symbol, signal.size)
        logger.info(f"预测滑点: {predicted_slippage:.5f}")
        
        # 调整价格
        adjusted_price = self._adjust_price_for_slippage(
            signal.entry_price,
            signal.direction,
            predicted_slippage
        )
        
        # 执行订单
        for attempt in range(self.max_retries):
            try:
                result = self._execute_order(signal, adjusted_price, attempt + 1)
                
                if result['success']:
                    # 记录实际滑点
                    actual_slippage = self._calculate_actual_slippage(
                        signal.entry_price,
                        result.get('fill_price', adjusted_price)
                    )
                    self._record_slippage(signal.symbol, actual_slippage)
                    
                    # 更新订单历史
                    self._add_to_history(result)
                    
                    logger.info(f"订单执行成功: {result.get('order_id')}, 实际滑点: {actual_slippage:.5f}")
                    return result
                else:
                    logger.warning(f"订单执行失败 (尝试 {attempt + 1}/{self.max_retries}): {result.get('error')}")
                    
                    # 检查是否应该重试
                    if not self._should_retry(result.get('error_code')):
                        break
                        
            except Exception as e:
                logger.error(f"订单执行异常 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt == self.max_retries - 1:
                    return self._create_order_result(
                        success=False,
                        error=f"订单执行异常: {str(e)}",
                        signal=signal
                    )
        
        return self._create_order_result(
            success=False,
            error="订单执行失败，已达最大重试次数",
            signal=signal
        )
    
    def _execute_order(self, signal: Signal, price: float, attempt: int) -> Dict[str, Any]:
        """
        执行订单（实际发送到MT5）
        
        Args:
            signal: 交易信号
            price: 调整后的价格
            attempt: 尝试次数
            
        Returns:
            执行结果
        """
        # 检查MT5连接
        if not mt5.initialize():
            raise ConnectionError("无法连接到MT5")
        
        # 准备订单请求
        request = self._prepare_order_request(signal, price)
        
        # 发送订单
        result = mt5.order_send(request)
        
        if result is None:
            error_code = mt5.last_error()
            return self._create_order_result(
                success=False,
                error=f"订单发送失败: {error_code}",
                error_code=error_code[0] if error_code else None,
                signal=signal
            )
        
        # 检查订单结果
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return self._create_order_result(
                success=False,
                error=f"订单未成交: {result.comment}",
                error_code=result.retcode,
                signal=signal,
                mt5_result=result
            )
        
        # 订单成功
        return self._create_order_result(
            success=True,
            order_id=str(result.order),
            fill_price=result.price,
            volume=result.volume,
            signal=signal,
            mt5_result=result
        )
    
    def _prepare_order_request(self, signal: Signal, price: float) -> Dict[str, Any]:
        """
        准备MT5订单请求
        
        Args:
            signal: 交易信号
            price: 执行价格
            
        Returns:
            MT5订单请求字典
        """
        # 确定订单类型
        if signal.direction == 1:  # 买入
            order_type = mt5.ORDER_TYPE_BUY
        elif signal.direction == -1:  # 卖出
            order_type = mt5.ORDER_TYPE_SELL
        else:
            raise ValueError(f"无效的交易方向: {signal.direction}")
        
        # 构建请求
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": signal.symbol,
            "volume": signal.size,
            "type": order_type,
            "price": price,
            "sl": signal.sl if signal.sl > 0 else 0.0,
            "tp": signal.tp if signal.tp > 0 else 0.0,
            "deviation": int(self.max_slippage * 10000),  # 转换为点数
            "magic": signal.metadata.get('magic_number', 0),
            "comment": f"Strategy: {signal.strategy_id}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        return request
    
    def predict_slippage(self, symbol: str, volume: float) -> float:
        """
        预测滑点
        
        Args:
            symbol: 交易品种
            volume: 交易量
            
        Returns:
            预测的滑点（百分比）
        """
        if self.slippage_model == SlippageModel.FIXED:
            return self.config.get('fixed_slippage', 0.0001)
        
        elif self.slippage_model == SlippageModel.HISTORICAL:
            # 基于历史滑点数据预测
            if symbol in self.slippage_history and len(self.slippage_history[symbol]) > 0:
                history = self.slippage_history[symbol]
                # 使用最近10次的平均值
                recent = history[-10:]
                avg_slippage = sum(recent) / len(recent)
                # 考虑交易量影响
                volume_factor = min(1.0 + (volume - 0.01) * 0.1, 2.0)
                return avg_slippage * volume_factor
            else:
                return self.config.get('default_slippage', 0.0002)
        
        else:  # DYNAMIC
            # 动态滑点模型：考虑市场波动和流动性
            return self._calculate_dynamic_slippage(symbol, volume)
    
    def _calculate_dynamic_slippage(self, symbol: str, volume: float) -> float:
        """
        计算动态滑点
        
        Args:
            symbol: 交易品种
            volume: 交易量
            
        Returns:
            动态滑点
        """
        try:
            # 获取当前市场信息
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return self.config.get('default_slippage', 0.0002)
            
            # 获取当前点差
            spread = symbol_info.spread
            point = symbol_info.point
            
            # 基础滑点 = 点差的一半
            base_slippage = (spread * point) / 2
            
            # 交易量因子
            volume_factor = 1.0 + (volume - 0.01) * 0.1
            
            # 时间因子（市场开盘/收盘时滑点更大）
            time_factor = self._get_time_factor()
            
            # 综合滑点
            dynamic_slippage = base_slippage * volume_factor * time_factor
            
            # 限制最大滑点
            return min(dynamic_slippage, self.max_slippage)
            
        except Exception as e:
            logger.warning(f"计算动态滑点失败: {str(e)}")
            return self.config.get('default_slippage', 0.0002)
    
    def _get_time_factor(self) -> float:
        """
        获取时间因子（考虑市场开盘/收盘时段）
        
        Returns:
            时间因子
        """
        current_hour = datetime.now().hour
        
        # 亚洲开盘 (0-2点) 和美国收盘 (22-24点) 滑点较大
        if current_hour in [0, 1, 2, 22, 23]:
            return 1.5
        # 欧洲开盘 (8-10点) 和美国开盘 (14-16点) 滑点中等
        elif current_hour in [8, 9, 10, 14, 15, 16]:
            return 1.2
        # 其他时段正常
        else:
            return 1.0
    
    def _adjust_price_for_slippage(self, price: float, direction: int, slippage: float) -> float:
        """
        根据滑点调整价格
        
        Args:
            price: 原始价格
            direction: 交易方向 (1=买入, -1=卖出)
            slippage: 预测的滑点
            
        Returns:
            调整后的价格
        """
        if direction == 1:  # 买入，价格可能更高
            return price * (1 + slippage)
        else:  # 卖出，价格可能更低
            return price * (1 - slippage)
    
    def _calculate_actual_slippage(self, expected_price: float, actual_price: float) -> float:
        """
        计算实际滑点
        
        Args:
            expected_price: 预期价格
            actual_price: 实际成交价格
            
        Returns:
            实际滑点（百分比）
        """
        if expected_price == 0:
            return 0.0
        return abs(actual_price - expected_price) / expected_price
    
    def _record_slippage(self, symbol: str, slippage: float) -> None:
        """
        记录滑点数据
        
        Args:
            symbol: 交易品种
            slippage: 滑点值
        """
        if symbol not in self.slippage_history:
            self.slippage_history[symbol] = []
        
        self.slippage_history[symbol].append(slippage)
        
        # 只保留最近100条记录
        if len(self.slippage_history[symbol]) > 100:
            self.slippage_history[symbol] = self.slippage_history[symbol][-100:]
    
    def _validate_signal(self, signal: Signal) -> bool:
        """
        验证交易信号
        
        Args:
            signal: 交易信号
            
        Returns:
            是否有效
        """
        # 检查基本参数
        if signal.size <= 0:
            logger.error("交易手数必须大于0")
            return False
        
        if signal.entry_price <= 0:
            logger.error("入场价格必须大于0")
            return False
        
        if not -1 <= signal.direction <= 1:
            logger.error("交易方向无效")
            return False
        
        # 检查止损止盈设置
        if signal.direction == 1:  # 买入
            if signal.sl > 0 and signal.sl >= signal.entry_price:
                logger.error("买入订单的止损价格必须低于入场价格")
                return False
            if signal.tp > 0 and signal.tp <= signal.entry_price:
                logger.error("买入订单的止盈价格必须高于入场价格")
                return False
        elif signal.direction == -1:  # 卖出
            if signal.sl > 0 and signal.sl <= signal.entry_price:
                logger.error("卖出订单的止损价格必须高于入场价格")
                return False
            if signal.tp > 0 and signal.tp >= signal.entry_price:
                logger.error("卖出订单的止盈价格必须低于入场价格")
                return False
        
        return True
    
    def _should_retry(self, error_code: Optional[int]) -> bool:
        """
        判断是否应该重试
        
        Args:
            error_code: 错误代码
            
        Returns:
            是否应该重试
        """
        # 不应重试的错误代码
        no_retry_codes = [
            10004,  # 资金不足
            10013,  # 无效手数
            10015,  # 无效价格
            10016,  # 无效止损止盈
            10021,  # 市场关闭
        ]
        
        if error_code in no_retry_codes:
            return False
        
        return True
    
    def _create_order_result(self, success: bool, signal: Signal, **kwargs) -> Dict[str, Any]:
        """
        创建订单结果字典
        
        Args:
            success: 是否成功
            signal: 交易信号
            **kwargs: 其他参数
            
        Returns:
            订单结果字典
        """
        result = {
            'success': success,
            'timestamp': datetime.now(),
            'symbol': signal.symbol,
            'direction': signal.direction,
            'volume': signal.size,
            'strategy_id': signal.strategy_id,
        }
        result.update(kwargs)
        return result
    
    def _add_to_history(self, result: Dict[str, Any]) -> None:
        """
        添加到订单历史
        
        Args:
            result: 订单结果
        """
        self.order_history.append(result)
        
        # 只保留最近1000条记录
        if len(self.order_history) > 1000:
            self.order_history = self.order_history[-1000:]
    
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        获取订单状态
        
        Args:
            order_id: 订单ID
            
        Returns:
            订单状态信息
        """
        # 从历史记录中查找
        for order in reversed(self.order_history):
            if order.get('order_id') == order_id:
                return order
        
        return None
    
    def get_slippage_statistics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        获取滑点统计信息
        
        Args:
            symbol: 交易品种（可选，None表示所有品种）
            
        Returns:
            滑点统计字典
        """
        if symbol:
            if symbol not in self.slippage_history or not self.slippage_history[symbol]:
                return {'symbol': symbol, 'count': 0}
            
            slippages = self.slippage_history[symbol]
            return {
                'symbol': symbol,
                'count': len(slippages),
                'avg': sum(slippages) / len(slippages),
                'min': min(slippages),
                'max': max(slippages),
                'recent_avg': sum(slippages[-10:]) / min(len(slippages), 10)
            }
        else:
            # 所有品种的统计
            stats = {}
            for sym, slippages in self.slippage_history.items():
                if slippages:
                    stats[sym] = {
                        'count': len(slippages),
                        'avg': sum(slippages) / len(slippages),
                        'min': min(slippages),
                        'max': max(slippages)
                    }
            return stats
    
    def cancel_pending_order(self, order_id: str) -> bool:
        """
        取消挂单
        
        Args:
            order_id: 订单ID
            
        Returns:
            是否成功
        """
        try:
            # 检查MT5连接
            if not mt5.initialize():
                raise ConnectionError("无法连接到MT5")
            
            # 准备取消请求
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": int(order_id),
            }
            
            # 发送取消请求
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"订单 {order_id} 已取消")
                return True
            else:
                logger.error(f"取消订单 {order_id} 失败: {result.comment if result else 'Unknown error'}")
                return False
                
        except Exception as e:
            logger.error(f"取消订单异常: {str(e)}")
            return False
