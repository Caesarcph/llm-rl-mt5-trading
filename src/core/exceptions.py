"""
异常处理基础框架
定义系统中使用的所有自定义异常类
"""

from typing import Optional, Dict, Any


class TradingSystemException(Exception):
    """交易系统基础异常类"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'exception_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'details': self.details
        }


class ConnectionException(TradingSystemException):
    """连接异常"""
    pass


# 为了向后兼容，添加别名
ConnectionError = ConnectionException


class MT5ConnectionException(ConnectionException):
    """MT5连接异常"""
    
    def __init__(self, message: str, mt5_error_code: Optional[int] = None):
        super().__init__(message, f"MT5_{mt5_error_code}" if mt5_error_code else None)
        self.mt5_error_code = mt5_error_code


class DatabaseException(TradingSystemException):
    """数据库异常"""
    pass


class DataException(TradingSystemException):
    """数据异常"""
    pass


class DataValidationException(DataException):
    """数据验证异常"""
    
    def __init__(self, message: str, field_name: Optional[str] = None, field_value: Optional[Any] = None):
        super().__init__(message, "DATA_VALIDATION")
        self.field_name = field_name
        self.field_value = field_value
        self.details.update({
            'field_name': field_name,
            'field_value': field_value
        })


# 为了向后兼容，添加别名
DataValidationError = DataValidationException


class DataProviderException(DataException):
    """数据提供者异常"""
    pass


class StrategyException(TradingSystemException):
    """策略异常"""
    pass


class IndicatorCalculationException(StrategyException):
    """指标计算异常"""
    
    def __init__(self, message: str, indicator_name: Optional[str] = None, symbol: Optional[str] = None):
        super().__init__(message, "INDICATOR_CALCULATION_ERROR")
        self.indicator_name = indicator_name
        self.symbol = symbol
        self.details.update({
            'indicator_name': indicator_name,
            'symbol': symbol
        })


# 为了向后兼容，添加别名
IndicatorCalculationError = IndicatorCalculationException


class AnalysisException(StrategyException):
    """分析异常"""
    
    def __init__(self, message: str, analysis_type: Optional[str] = None, symbol: Optional[str] = None):
        super().__init__(message, "ANALYSIS_ERROR")
        self.analysis_type = analysis_type
        self.symbol = symbol
        self.details.update({
            'analysis_type': analysis_type,
            'symbol': symbol
        })


# 为了向后兼容，添加别名
AnalysisError = AnalysisException


class SignalException(StrategyException):
    """信号异常"""
    
    def __init__(self, message: str, signal_id: Optional[str] = None, strategy_id: Optional[str] = None):
        super().__init__(message, "SIGNAL_ERROR")
        self.signal_id = signal_id
        self.strategy_id = strategy_id
        self.details.update({
            'signal_id': signal_id,
            'strategy_id': strategy_id
        })


class OrderException(TradingSystemException):
    """订单异常"""
    
    def __init__(self, message: str, order_id: Optional[str] = None, mt5_error_code: Optional[int] = None):
        super().__init__(message, f"ORDER_{mt5_error_code}" if mt5_error_code else "ORDER_ERROR")
        self.order_id = order_id
        self.mt5_error_code = mt5_error_code
        self.details.update({
            'order_id': order_id,
            'mt5_error_code': mt5_error_code
        })


class OrderExecutionError(OrderException):
    """订单执行错误"""
    pass


# 为了向后兼容，添加别名
OrderExecutionException = OrderExecutionError


class InsufficientFundsException(OrderException):
    """资金不足异常"""
    
    def __init__(self, required_margin: float, available_margin: float):
        message = f"资金不足: 需要 {required_margin}, 可用 {available_margin}"
        super().__init__(message, error_code="INSUFFICIENT_FUNDS")
        self.required_margin = required_margin
        self.available_margin = available_margin
        self.details.update({
            'required_margin': required_margin,
            'available_margin': available_margin
        })


class InvalidOrderException(OrderException):
    """无效订单异常"""
    pass


class RiskException(TradingSystemException):
    """风险管理异常"""
    pass


class RiskLimitExceededException(RiskException):
    """风险限制超出异常"""
    
    def __init__(self, risk_type: str, current_value: float, limit_value: float):
        message = f"{risk_type}超出限制: 当前 {current_value}, 限制 {limit_value}"
        super().__init__(message, "RISK_LIMIT_EXCEEDED")
        self.risk_type = risk_type
        self.current_value = current_value
        self.limit_value = limit_value
        self.details.update({
            'risk_type': risk_type,
            'current_value': current_value,
            'limit_value': limit_value
        })


class MaxDrawdownException(RiskLimitExceededException):
    """最大回撤异常"""
    
    def __init__(self, current_drawdown: float, max_drawdown: float):
        super().__init__("最大回撤", current_drawdown, max_drawdown)


class PositionLimitException(RiskLimitExceededException):
    """持仓限制异常"""
    
    def __init__(self, current_positions: int, max_positions: int):
        super().__init__("持仓数量", current_positions, max_positions)


class ConfigurationException(TradingSystemException):
    """配置异常"""
    pass


class BridgeError(TradingSystemException):
    """桥接器异常"""
    pass


class InvalidConfigException(ConfigurationException):
    """无效配置异常"""
    
    def __init__(self, config_key: str, config_value: Any, reason: str):
        message = f"无效配置 {config_key}={config_value}: {reason}"
        super().__init__(message, "INVALID_CONFIG")
        self.config_key = config_key
        self.config_value = config_value
        self.reason = reason
        self.details.update({
            'config_key': config_key,
            'config_value': config_value,
            'reason': reason
        })


class LLMException(TradingSystemException):
    """LLM异常"""
    pass


class ModelLoadException(LLMException):
    """模型加载异常"""
    
    def __init__(self, model_path: str, reason: str):
        message = f"模型加载失败 {model_path}: {reason}"
        super().__init__(message, "MODEL_LOAD_ERROR")
        self.model_path = model_path
        self.reason = reason
        self.details.update({
            'model_path': model_path,
            'reason': reason
        })


class InferenceException(LLMException):
    """推理异常"""
    pass


class RLException(TradingSystemException):
    """强化学习异常"""
    pass


class EnvironmentException(RLException):
    """环境异常"""
    pass


class TrainingException(RLException):
    """训练异常"""
    pass


class BacktestException(TradingSystemException):
    """回测异常"""
    pass


class OptimizationException(TradingSystemException):
    """优化异常"""
    pass


# 异常处理装饰器
def handle_exceptions(exception_types: tuple = (Exception,), 
                     default_return=None, 
                     log_error: bool = True,
                     reraise: bool = False):
    """异常处理装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_types as e:
                if log_error:
                    from .logging import log_error
                    log_error(f"函数 {func.__name__} 执行异常", e)
                
                if reraise:
                    raise
                
                return default_return
        return wrapper
    return decorator


def retry_on_exception(max_retries: int = 3, 
                      delay: float = 1.0, 
                      exception_types: tuple = (Exception,),
                      backoff_factor: float = 2.0):
    """重试装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exception_types as e:
                    last_exception = e
                    if attempt < max_retries:
                        from .logging import get_logger
                        logger = get_logger()
                        logger.warning(f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}, {current_delay}秒后重试")
                        
                        import time
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        break
            
            # 所有重试都失败，抛出最后一个异常
            raise last_exception
        return wrapper
    return decorator


class ExceptionHandler:
    """异常处理器"""
    
    def __init__(self):
        self.handlers = {}
    
    def register_handler(self, exception_type: type, handler_func):
        """注册异常处理器"""
        self.handlers[exception_type] = handler_func
    
    def handle_exception(self, exception: Exception) -> Any:
        """处理异常"""
        exception_type = type(exception)
        
        # 查找精确匹配的处理器
        if exception_type in self.handlers:
            return self.handlers[exception_type](exception)
        
        # 查找父类匹配的处理器
        for registered_type, handler in self.handlers.items():
            if isinstance(exception, registered_type):
                return handler(exception)
        
        # 没有找到处理器，重新抛出异常
        raise exception


# 全局异常处理器
global_exception_handler = ExceptionHandler()


def register_exception_handler(exception_type: type, handler_func):
    """注册全局异常处理器"""
    global_exception_handler.register_handler(exception_type, handler_func)


def handle_exception(exception: Exception) -> Any:
    """处理异常"""
    return global_exception_handler.handle_exception(exception)