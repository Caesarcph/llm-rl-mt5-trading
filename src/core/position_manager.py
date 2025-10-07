"""
仓位管理器模块
管理所有持仓、追踪止损、部分平仓和风险监控
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import MetaTrader5 as mt5

from src.core.models import Position, PositionType, Account
from src.core.exceptions import (
    ConnectionError, 
    OrderException, 
    RiskLimitExceededException,
    PositionLimitException
)


logger = logging.getLogger(__name__)


class TrailingStopType(Enum):
    """追踪止损类型"""
    FIXED = "fixed"  # 固定点数
    PERCENTAGE = "percentage"  # 百分比
    ATR = "atr"  # ATR倍数


class PositionManager:
    """仓位管理器类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化仓位管理器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.max_positions = self.config.get('max_positions', 10)
        self.max_risk_per_position = self.config.get('max_risk_per_position', 0.02)  # 2%
        self.trailing_stop_enabled = self.config.get('trailing_stop_enabled', True)
        self.partial_close_enabled = self.config.get('partial_close_enabled', True)
        
        # 仓位跟踪
        self.positions: Dict[str, Position] = {}
        self.position_history: List[Dict[str, Any]] = []
        
        # 追踪止损配置
        self.trailing_stops: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"仓位管理器初始化完成，最大持仓数: {self.max_positions}")
    
    def update_positions(self) -> List[Position]:
        """
        从MT5更新所有持仓
        
        Returns:
            持仓列表
        """
        try:
            if not mt5.initialize():
                raise ConnectionError("无法连接到MT5")
            
            # 获取所有持仓
            mt5_positions = mt5.positions_get()
            
            if mt5_positions is None:
                logger.warning("获取持仓失败")
                return []
            
            # 更新内部持仓字典
            current_position_ids = set()
            
            for mt5_pos in mt5_positions:
                position_id = str(mt5_pos.ticket)
                current_position_ids.add(position_id)
                
                # 转换为内部Position对象
                position = self._convert_mt5_position(mt5_pos)
                
                # 更新或添加持仓
                if position_id in self.positions:
                    self.positions[position_id].update_current_price(position.current_price)
                    self.positions[position_id].profit = position.profit
                else:
                    self.positions[position_id] = position
                    logger.info(f"新增持仓: {position_id} {position.symbol} {position.type.name}")
            
            # 移除已关闭的持仓
            closed_positions = set(self.positions.keys()) - current_position_ids
            for pos_id in closed_positions:
                logger.info(f"持仓已关闭: {pos_id}")
                self._archive_position(self.positions[pos_id])
                del self.positions[pos_id]
                if pos_id in self.trailing_stops:
                    del self.trailing_stops[pos_id]
            
            return list(self.positions.values())
            
        except Exception as e:
            logger.error(f"更新持仓失败: {str(e)}")
            return []
    
    def _convert_mt5_position(self, mt5_pos) -> Position:
        """
        转换MT5持仓为内部Position对象
        
        Args:
            mt5_pos: MT5持仓对象
            
        Returns:
            Position对象
        """
        position_type = PositionType.LONG if mt5_pos.type == mt5.POSITION_TYPE_BUY else PositionType.SHORT
        
        return Position(
            position_id=str(mt5_pos.ticket),
            symbol=mt5_pos.symbol,
            type=position_type,
            volume=mt5_pos.volume,
            open_price=mt5_pos.price_open,
            current_price=mt5_pos.price_current,
            sl=mt5_pos.sl,
            tp=mt5_pos.tp,
            profit=mt5_pos.profit,
            swap=mt5_pos.swap,
            commission=mt5_pos.commission if hasattr(mt5_pos, 'commission') else 0.0,
            open_time=datetime.fromtimestamp(mt5_pos.time),
            comment=mt5_pos.comment if hasattr(mt5_pos, 'comment') else "",
            magic_number=mt5_pos.magic
        )
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """
        获取指定持仓
        
        Args:
            position_id: 持仓ID
            
        Returns:
            Position对象或None
        """
        return self.positions.get(position_id)
    
    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """
        获取指定品种的所有持仓
        
        Args:
            symbol: 交易品种
            
        Returns:
            持仓列表
        """
        return [pos for pos in self.positions.values() if pos.symbol == symbol]
    
    def get_total_exposure(self, symbol: Optional[str] = None) -> float:
        """
        获取总敞口
        
        Args:
            symbol: 交易品种（可选）
            
        Returns:
            总敞口（手数）
        """
        if symbol:
            positions = self.get_positions_by_symbol(symbol)
        else:
            positions = list(self.positions.values())
        
        return sum(pos.volume for pos in positions)
    
    def get_net_exposure(self, symbol: str) -> float:
        """
        获取净敞口（多头-空头）
        
        Args:
            symbol: 交易品种
            
        Returns:
            净敞口
        """
        positions = self.get_positions_by_symbol(symbol)
        
        long_volume = sum(pos.volume for pos in positions if pos.type == PositionType.LONG)
        short_volume = sum(pos.volume for pos in positions if pos.type == PositionType.SHORT)
        
        return long_volume - short_volume
    
    def close_position(self, position_id: str, volume: Optional[float] = None) -> bool:
        """
        关闭持仓（全部或部分）
        
        Args:
            position_id: 持仓ID
            volume: 关闭手数（None表示全部关闭）
            
        Returns:
            是否成功
        """
        position = self.get_position(position_id)
        if not position:
            logger.error(f"持仓不存在: {position_id}")
            return False
        
        try:
            if not mt5.initialize():
                raise ConnectionError("无法连接到MT5")
            
            # 确定关闭手数
            close_volume = volume if volume else position.volume
            
            if close_volume > position.volume:
                logger.error(f"关闭手数 {close_volume} 超过持仓手数 {position.volume}")
                return False
            
            # 准备平仓请求
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": int(position_id),
                "symbol": position.symbol,
                "volume": close_volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == PositionType.LONG else mt5.ORDER_TYPE_BUY,
                "price": position.current_price,
                "deviation": 20,
                "magic": position.magic_number,
                "comment": f"Close position {position_id}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # 发送平仓请求
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                logger.error(f"平仓失败: {error}")
                return False
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"平仓未成交: {result.comment}")
                return False
            
            logger.info(f"持仓 {position_id} 平仓成功，手数: {close_volume}")
            
            # 如果是部分平仓，更新持仓信息
            if close_volume < position.volume:
                self.update_positions()
            
            return True
            
        except Exception as e:
            logger.error(f"平仓异常: {str(e)}")
            return False
    
    def modify_position(self, position_id: str, sl: Optional[float] = None, 
                       tp: Optional[float] = None) -> bool:
        """
        修改持仓的止损止盈
        
        Args:
            position_id: 持仓ID
            sl: 新的止损价格（None表示不修改）
            tp: 新的止盈价格（None表示不修改）
            
        Returns:
            是否成功
        """
        position = self.get_position(position_id)
        if not position:
            logger.error(f"持仓不存在: {position_id}")
            return False
        
        try:
            if not mt5.initialize():
                raise ConnectionError("无法连接到MT5")
            
            # 使用当前值如果没有提供新值
            new_sl = sl if sl is not None else position.sl
            new_tp = tp if tp is not None else position.tp
            
            # 准备修改请求
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": int(position_id),
                "symbol": position.symbol,
                "sl": new_sl,
                "tp": new_tp,
            }
            
            # 发送修改请求
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                logger.error(f"修改持仓失败: {error}")
                return False
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"修改持仓未成功: {result.comment}")
                return False
            
            logger.info(f"持仓 {position_id} 修改成功，SL: {new_sl}, TP: {new_tp}")
            
            # 更新内部持仓信息
            position.sl = new_sl
            position.tp = new_tp
            
            return True
            
        except Exception as e:
            logger.error(f"修改持仓异常: {str(e)}")
            return False
    
    def enable_trailing_stop(self, position_id: str, 
                           trailing_type: TrailingStopType = TrailingStopType.FIXED,
                           trailing_value: float = 50.0,
                           activation_profit: float = 0.0) -> bool:
        """
        启用追踪止损
        
        Args:
            position_id: 持仓ID
            trailing_type: 追踪止损类型
            trailing_value: 追踪值（点数、百分比或ATR倍数）
            activation_profit: 激活盈利（只有盈利超过此值才启动追踪）
            
        Returns:
            是否成功
        """
        position = self.get_position(position_id)
        if not position:
            logger.error(f"持仓不存在: {position_id}")
            return False
        
        self.trailing_stops[position_id] = {
            'type': trailing_type,
            'value': trailing_value,
            'activation_profit': activation_profit,
            'highest_price': position.current_price if position.type == PositionType.LONG else None,
            'lowest_price': position.current_price if position.type == PositionType.SHORT else None,
            'activated': False
        }
        
        logger.info(f"持仓 {position_id} 启用追踪止损: {trailing_type.value}, 值: {trailing_value}")
        return True
    
    def update_trailing_stops(self) -> None:
        """更新所有追踪止损"""
        if not self.trailing_stop_enabled:
            return
        
        for position_id, trailing_config in list(self.trailing_stops.items()):
            position = self.get_position(position_id)
            if not position:
                continue
            
            # 检查是否达到激活条件
            if not trailing_config['activated']:
                if position.profit >= trailing_config['activation_profit']:
                    trailing_config['activated'] = True
                    logger.info(f"持仓 {position_id} 追踪止损已激活")
                else:
                    continue
            
            # 计算新的止损价格
            new_sl = self._calculate_trailing_stop(position, trailing_config)
            
            if new_sl is None:
                continue
            
            # 检查是否需要更新止损
            should_update = False
            
            if position.type == PositionType.LONG:
                # 多头：新止损高于当前止损
                if position.sl == 0 or new_sl > position.sl:
                    should_update = True
            else:
                # 空头：新止损低于当前止损
                if position.sl == 0 or new_sl < position.sl:
                    should_update = True
            
            if should_update:
                if self.modify_position(position_id, sl=new_sl):
                    logger.info(f"持仓 {position_id} 追踪止损更新: {new_sl}")
    
    def _calculate_trailing_stop(self, position: Position, 
                                 trailing_config: Dict[str, Any]) -> Optional[float]:
        """
        计算追踪止损价格
        
        Args:
            position: 持仓对象
            trailing_config: 追踪配置
            
        Returns:
            新的止损价格
        """
        trailing_type = trailing_config['type']
        trailing_value = trailing_config['value']
        
        if position.type == PositionType.LONG:
            # 更新最高价
            if trailing_config['highest_price'] is None or position.current_price > trailing_config['highest_price']:
                trailing_config['highest_price'] = position.current_price
            
            highest = trailing_config['highest_price']
            
            if trailing_type == TrailingStopType.FIXED:
                # 固定点数
                symbol_info = mt5.symbol_info(position.symbol)
                if symbol_info:
                    point = symbol_info.point
                    return highest - (trailing_value * point)
            
            elif trailing_type == TrailingStopType.PERCENTAGE:
                # 百分比
                return highest * (1 - trailing_value / 100)
        
        else:  # SHORT
            # 更新最低价
            if trailing_config['lowest_price'] is None or position.current_price < trailing_config['lowest_price']:
                trailing_config['lowest_price'] = position.current_price
            
            lowest = trailing_config['lowest_price']
            
            if trailing_type == TrailingStopType.FIXED:
                # 固定点数
                symbol_info = mt5.symbol_info(position.symbol)
                if symbol_info:
                    point = symbol_info.point
                    return lowest + (trailing_value * point)
            
            elif trailing_type == TrailingStopType.PERCENTAGE:
                # 百分比
                return lowest * (1 + trailing_value / 100)
        
        return None
    
    def partial_close_position(self, position_id: str, percentage: float) -> bool:
        """
        部分平仓
        
        Args:
            position_id: 持仓ID
            percentage: 平仓百分比（0-100）
            
        Returns:
            是否成功
        """
        if not self.partial_close_enabled:
            logger.warning("部分平仓功能未启用")
            return False
        
        if not 0 < percentage < 100:
            logger.error(f"无效的平仓百分比: {percentage}")
            return False
        
        position = self.get_position(position_id)
        if not position:
            logger.error(f"持仓不存在: {position_id}")
            return False
        
        # 计算平仓手数
        close_volume = position.volume * (percentage / 100)
        
        # 获取品种信息以调整手数
        symbol_info = mt5.symbol_info(position.symbol)
        if symbol_info:
            # 调整到最小步长
            close_volume = round(close_volume / symbol_info.volume_step) * symbol_info.volume_step
            close_volume = max(symbol_info.volume_min, min(close_volume, position.volume))
        
        logger.info(f"部分平仓 {position_id}: {percentage}% = {close_volume}手")
        return self.close_position(position_id, close_volume)
    
    def monitor_position_risk(self, position: Position, account: Account) -> Dict[str, Any]:
        """
        监控单个持仓风险
        
        Args:
            position: 持仓对象
            account: 账户对象
            
        Returns:
            风险指标字典
        """
        # 计算风险指标
        risk_amount = abs(position.profit) if position.profit < 0 else 0
        risk_percentage = (risk_amount / account.balance) * 100 if account.balance > 0 else 0
        
        # 计算潜在最大损失（如果有止损）
        max_loss = 0
        if position.sl > 0:
            if position.type == PositionType.LONG:
                max_loss = (position.open_price - position.sl) * position.volume
            else:
                max_loss = (position.sl - position.open_price) * position.volume
        
        max_loss_percentage = (max_loss / account.balance) * 100 if account.balance > 0 else 0
        
        # 持仓时长
        duration_hours = (datetime.now() - position.open_time).total_seconds() / 3600
        
        return {
            'position_id': position.position_id,
            'symbol': position.symbol,
            'current_profit': position.profit,
            'risk_amount': risk_amount,
            'risk_percentage': risk_percentage,
            'max_loss': max_loss,
            'max_loss_percentage': max_loss_percentage,
            'duration_hours': duration_hours,
            'has_sl': position.sl > 0,
            'has_tp': position.tp > 0
        }
    
    def get_portfolio_risk_metrics(self, account: Account) -> Dict[str, Any]:
        """
        获取组合风险指标
        
        Args:
            account: 账户对象
            
        Returns:
            组合风险指标
        """
        if not self.positions:
            return {
                'total_positions': 0,
                'total_profit': 0,
                'total_risk': 0,
                'risk_percentage': 0,
                'largest_position_risk': 0
            }
        
        total_profit = sum(pos.profit for pos in self.positions.values())
        total_risk = sum(abs(pos.profit) for pos in self.positions.values() if pos.profit < 0)
        
        # 计算每个持仓的风险
        position_risks = []
        for pos in self.positions.values():
            risk_metrics = self.monitor_position_risk(pos, account)
            position_risks.append(risk_metrics['risk_percentage'])
        
        return {
            'total_positions': len(self.positions),
            'total_profit': total_profit,
            'total_risk': total_risk,
            'risk_percentage': (total_risk / account.balance * 100) if account.balance > 0 else 0,
            'largest_position_risk': max(position_risks) if position_risks else 0,
            'avg_position_risk': sum(position_risks) / len(position_risks) if position_risks else 0
        }
    
    def check_position_limits(self) -> Tuple[bool, Optional[str]]:
        """
        检查持仓限制
        
        Returns:
            (是否在限制内, 错误信息)
        """
        if len(self.positions) >= self.max_positions:
            return False, f"持仓数量已达上限: {self.max_positions}"
        
        return True, None
    
    def _archive_position(self, position: Position) -> None:
        """
        归档已关闭的持仓
        
        Args:
            position: 持仓对象
        """
        self.position_history.append({
            'position_id': position.position_id,
            'symbol': position.symbol,
            'type': position.type.name,
            'volume': position.volume,
            'open_price': position.open_price,
            'close_price': position.current_price,
            'profit': position.profit,
            'open_time': position.open_time,
            'close_time': datetime.now(),
            'duration_hours': (datetime.now() - position.open_time).total_seconds() / 3600
        })
        
        # 只保留最近1000条记录
        if len(self.position_history) > 1000:
            self.position_history = self.position_history[-1000:]
    
    def get_position_statistics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        获取持仓统计信息
        
        Args:
            symbol: 交易品种（可选）
            
        Returns:
            统计信息字典
        """
        history = self.position_history
        if symbol:
            history = [h for h in history if h['symbol'] == symbol]
        
        if not history:
            return {'count': 0}
        
        profits = [h['profit'] for h in history]
        durations = [h['duration_hours'] for h in history]
        
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        return {
            'count': len(history),
            'total_profit': sum(profits),
            'avg_profit': sum(profits) / len(profits),
            'win_rate': len(winning_trades) / len(profits) * 100 if profits else 0,
            'avg_win': sum(winning_trades) / len(winning_trades) if winning_trades else 0,
            'avg_loss': sum(losing_trades) / len(losing_trades) if losing_trades else 0,
            'avg_duration': sum(durations) / len(durations),
            'max_profit': max(profits),
            'max_loss': min(profits)
        }
