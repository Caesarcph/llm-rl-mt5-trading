# LLM-RL MT5 Trading System 部署指南

## 目录
1. [系统要求](#系统要求)
2. [安装步骤](#安装步骤)
3. [配置说明](#配置说明)
4. [部署选项](#部署选项)
5. [故障排除](#故障排除)

## 系统要求

### 最低配置
- **操作系统**: Windows 10/11, Linux (Ubuntu 20.04+), macOS 11+
- **Python**: 3.9 或更高版本
- **CPU**: 2核心
- **内存**: 4GB RAM
- **存储**: 10GB 可用空间
- **网络**: 稳定的互联网连接

### 推荐配置
- **操作系统**: Windows 11, Ubuntu 22.04
- **Python**: 3.10+
- **CPU**: 4核心或更多
- **内存**: 8GB RAM 或更多
- **存储**: 20GB SSD
- **GPU**: NVIDIA GPU (用于LLM加速，可选)

### 软件依赖
- MetaTrader 5 平台
- Redis (可选，用于缓存)
- Git (用于版本控制)

## 安装步骤

### 1. 克隆或下载项目

```bash
# 使用Git克隆
git clone <repository-url>
cd llm-rl-mt5-trading

# 或直接下载并解压
```

### 2. 运行安装脚本

#### Windows
```cmd
python install.py
```

#### Linux/macOS
```bash
python3 install.py
```

安装脚本将自动:
- 检查Python版本
- 创建必要的目录结构
- 安装所有依赖包
- 配置系统
- 验证安装

### 3. 手动安装 (可选)

如果自动安装失败，可以手动执行以下步骤:

```bash
# 1. 创建虚拟环境 (推荐)
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 创建目录
python -c "from pathlib import Path; [Path(d).mkdir(parents=True, exist_ok=True) for d in ['data', 'logs', 'models', 'ea31337']]"

# 4. 验证安装
python -c "from src.core.environment import check_environment; check_environment()"
```

## 配置说明

### 1. 主配置文件 (config/config.yaml)

编辑 `config/config.yaml` 文件，配置以下关键参数:

#### MT5连接配置
```yaml
mt5:
  server: "YourBroker-Server"    # MT5服务器地址
  login: 12345678                # MT5账号
  password: "your_password"      # MT5密码
  timeout: 60000
```

#### 风险管理配置
```yaml
risk:
  max_risk_per_trade: 0.02      # 单笔最大风险2%
  max_daily_drawdown: 0.05      # 日最大回撤5%
  max_positions: 10             # 最大持仓数
```

#### 交易品种配置
```yaml
trading:
  symbols: ["EURUSD", "XAUUSD"]  # 交易品种
  default_lot_size: 0.01         # 默认手数
```

#### 系统模式
```yaml
simulation_mode: true           # true=模拟模式, false=实盘模式
debug_mode: false              # 调试模式
```

### 2. 告警配置 (config/alert_config.yaml)

配置Telegram或邮件告警:

```yaml
telegram:
  enabled: true
  bot_token: "your_bot_token"
  chat_id: "your_chat_id"

email:
  enabled: false
  smtp_server: "smtp.gmail.com"
  smtp_port: 587
  username: "your_email@gmail.com"
  password: "your_password"
```

### 3. LLM配置 (config/llm_config.yaml)

配置本地LLM模型:

```yaml
model:
  path: "models/llama-3.2-1b"
  type: "llama"
  use_gpu: true
```

## 部署选项

### 选项1: 本地部署 (推荐用于开发和测试)

**优点:**
- 完全控制
- 低延迟
- 无额外成本

**步骤:**
1. 在本地机器上完成安装
2. 确保MT5已安装并运行
3. 运行启动脚本:
   ```bash
   # Windows
   start.bat
   
   # Linux/macOS
   ./start.sh
   ```

### 选项2: VPS部署 (推荐用于生产)

**优点:**
- 24/7运行
- 稳定网络
- 低延迟连接到MT5服务器

**推荐VPS配置:**
- 2核CPU, 4GB RAM (最低)
- 4核CPU, 8GB RAM (推荐)
- Windows Server 2019+ 或 Ubuntu 20.04+
- 位置: 靠近MT5服务器所在地区

**部署步骤:**

1. **连接到VPS**
   ```bash
   # Linux
   ssh user@your-vps-ip
   
   # Windows: 使用远程桌面连接
   ```

2. **安装必要软件**
   ```bash
   # Ubuntu
   sudo apt update
   sudo apt install python3.10 python3-pip git redis-server
   
   # Windows: 手动安装Python, Git, Redis
   ```

3. **上传项目文件**
   ```bash
   # 使用Git
   git clone <repository-url>
   
   # 或使用SCP/FTP上传
   ```

4. **运行安装脚本**
   ```bash
   python3 install.py
   ```

5. **配置自动启动**
   
   **Linux (systemd):**
   创建 `/etc/systemd/system/trading-system.service`:
   ```ini
   [Unit]
   Description=LLM-RL MT5 Trading System
   After=network.target
   
   [Service]
   Type=simple
   User=your_user
   WorkingDirectory=/path/to/project
   ExecStart=/usr/bin/python3 /path/to/project/main.py
   Restart=always
   RestartSec=10
   
   [Install]
   WantedBy=multi-user.target
   ```
   
   启用服务:
   ```bash
   sudo systemctl enable trading-system
   sudo systemctl start trading-system
   sudo systemctl status trading-system
   ```
   
   **Windows (任务计划程序):**
   1. 打开任务计划程序
   2. 创建基本任务
   3. 触发器: 系统启动时
   4. 操作: 启动程序 `python.exe main.py`

### 选项3: Docker部署 (高级)

**优点:**
- 环境隔离
- 易于迁移
- 一致性

**Dockerfile示例:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

**部署命令:**
```bash
# 构建镜像
docker build -t trading-system .

# 运行容器
docker run -d \
  --name trading-system \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  --restart unless-stopped \
  trading-system
```

## 验证部署

### 1. 检查系统状态

```bash
# 查看日志
tail -f logs/system.log

# 检查进程
ps aux | grep python
```

### 2. 测试MT5连接

```python
python -c "
from src.data.mt5_connection import MT5Connection, ConnectionConfig
config = ConnectionConfig()
conn = MT5Connection(config)
print('连接成功' if conn.connect() else '连接失败')
"
```

### 3. 运行测试套件

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_trading_system.py
```

## 监控和维护

### 日志文件位置
- 系统日志: `logs/system.log`
- 交易日志: `logs/trading.log`
- 错误日志: `logs/error.log`
- 策略日志: `logs/strategy.log`

### 监控指标
- 系统运行时间
- 交易成功率
- 当前回撤
- 错误计数
- MT5连接状态

### 定期维护任务
1. **每日**: 检查日志文件，查看异常
2. **每周**: 备份数据库和配置文件
3. **每月**: 更新依赖包，检查系统性能
4. **季度**: 审查策略表现，优化参数

## 故障排除

### 问题1: MT5连接失败

**症状**: 日志显示 "MT5连接失败"

**解决方案**:
1. 确认MT5已安装并运行
2. 检查config.yaml中的MT5配置
3. 验证账号密码正确
4. 检查网络连接
5. 尝试手动登录MT5

### 问题2: 依赖包安装失败

**症状**: pip install 报错

**解决方案**:
```bash
# 升级pip
python -m pip install --upgrade pip

# 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 单独安装失败的包
pip install package_name --no-cache-dir
```

### 问题3: 内存不足

**症状**: 系统崩溃或变慢

**解决方案**:
1. 减少同时运行的策略数量
2. 降低数据缓存大小
3. 使用更小的LLM模型 (1B而非3B)
4. 增加系统内存

### 问题4: 权限错误

**症状**: 无法创建文件或目录

**解决方案**:
```bash
# Linux/macOS
chmod -R 755 /path/to/project
chown -R user:user /path/to/project

# Windows: 右键 -> 属性 -> 安全 -> 编辑权限
```

### 问题5: Redis连接失败

**症状**: 缓存功能不可用

**解决方案**:
```bash
# 检查Redis是否运行
# Linux
sudo systemctl status redis

# 启动Redis
sudo systemctl start redis

# Windows: 启动Redis服务
```

## 安全建议

1. **不要在公共仓库中提交配置文件**
   ```bash
   # 添加到.gitignore
   config/config.yaml
   config/*_config.yaml
   ```

2. **使用环境变量存储敏感信息**
   ```bash
   export MT5_PASSWORD="your_password"
   export TELEGRAM_TOKEN="your_token"
   ```

3. **定期更新密码和API密钥**

4. **限制VPS访问**
   - 使用SSH密钥认证
   - 配置防火墙
   - 禁用root登录

5. **加密敏感数据**
   - 数据库加密
   - 配置文件加密

## 性能优化

1. **使用SSD存储**
2. **启用Redis缓存**
3. **优化数据库查询**
4. **使用GPU加速LLM推理**
5. **调整系统参数**:
   ```yaml
   system:
     loop_interval: 1  # 主循环间隔(秒)
     monitoring_interval: 30  # 监控间隔(秒)
   ```

## 支持和帮助

- 文档: `docs/` 目录
- 示例: `examples/` 目录
- 测试: `tests/` 目录
- 日志: `logs/` 目录

## 更新系统

```bash
# 备份当前配置
cp -r config config.backup

# 拉取最新代码
git pull origin main

# 更新依赖
pip install -r requirements.txt --upgrade

# 重启系统
```

## 卸载

```bash
# 停止系统
# Linux
sudo systemctl stop trading-system
sudo systemctl disable trading-system

# 删除文件
rm -rf /path/to/project

# 删除虚拟环境
rm -rf venv
```
