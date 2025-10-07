# 安装和配置指南

## 概述

本指南提供LLM-RL MT5交易系统的详细安装步骤，适用于Windows和Linux系统。

## 目录

- [Windows安装](#windows安装)
- [Linux安装](#linux安装)
- [Docker部署](#docker部署)
- [配置步骤](#配置步骤)
- [验证安装](#验证安装)

---

## Windows安装

### 前置要求

- Windows 10/11 (64位)
- 管理员权限
- 稳定的互联网连接

### 步骤1: 安装Python

1. 访问 https://www.python.org/downloads/
2. 下载Python 3.9或更高版本
3. 运行安装程序
4. **重要**: 勾选 "Add Python to PATH"
5. 点击 "Install Now"

验证安装：
```cmd
python --version
pip --version
```

### 步骤2: 安装Git

1. 访问 https://git-scm.com/download/win
2. 下载并安装Git
3. 使用默认设置

### 步骤3: 安装MetaTrader 5

1. 访问您的经纪商网站
2. 下载MT5平台
3. 安装到默认位置: `C:\Program Files\MetaTrader 5\`
4. 登录您的交易账户

### 步骤4: 安装Redis

**方法1: 使用Memurai (推荐)**
1. 访问 https://www.memurai.com/
2. 下载Memurai (Redis for Windows)
3. 安装并启动服务

**方法2: 使用WSL**
```bash
# 在WSL中安装Redis
wsl --install
wsl
sudo apt update
sudo apt install redis-server
sudo service redis-server start
```

验证Redis：
```cmd
redis-cli ping
```

### 步骤5: 下载交易系统

```cmd
# 克隆仓库
git clone https://github.com/your-repo/llm-rl-mt5-trading.git
cd llm-rl-mt5-trading

# 或下载ZIP并解压
```

### 步骤6: 创建虚拟环境

```cmd
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
venv\Scripts\activate

# 升级pip
python -m pip install --upgrade pip
```

### 步骤7: 安装依赖

```cmd
# 安装Python依赖
pip install -r requirements.txt

# 如果遇到ta-lib安装问题
# 下载预编译版本: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.24-cp39-cp39-win_amd64.whl
```

### 步骤8: 下载LLM模型（可选）

```cmd
# 安装Hugging Face CLI
pip install huggingface-hub

# 下载模型
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir models/llama-3.2-1b
```

---

## Linux安装

### 前置要求

- Ubuntu 20.04+ 或其他Linux发行版
- sudo权限
- 稳定的互联网连接

### 步骤1: 更新系统

```bash
sudo apt update
sudo apt upgrade -y
```

### 步骤2: 安装Python和依赖

```bash
# 安装Python 3.9
sudo apt install python3.9 python3.9-venv python3-pip -y

# 安装系统依赖
sudo apt install build-essential libssl-dev libffi-dev python3-dev -y

# 验证安装
python3.9 --version
```

### 步骤3: 安装Git

```bash
sudo apt install git -y
```

### 步骤4: 安装Redis

```bash
# 安装Redis
sudo apt install redis-server -y

# 启动Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# 验证
redis-cli ping
```

### 步骤5: 安装MetaTrader 5

**方法1: 使用Wine (推荐)**
```bash
# 安装Wine
sudo dpkg --add-architecture i386
sudo apt update
sudo apt install wine64 wine32 -y

# 下载MT5
wget https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe

# 安装MT5
wine mt5setup.exe
```

**方法2: 使用VPS**
- 使用Windows VPS运行MT5
- Python系统运行在Linux服务器
- 通过网络连接

### 步骤6: 下载交易系统

```bash
# 克隆仓库
git clone https://github.com/your-repo/llm-rl-mt5-trading.git
cd llm-rl-mt5-trading
```

### 步骤7: 创建虚拟环境

```bash
# 创建虚拟环境
python3.9 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 升级pip
pip install --upgrade pip
```

### 步骤8: 安装依赖

```bash
# 安装ta-lib依赖
sudo apt install libta-lib0-dev -y

# 安装Python依赖
pip install -r requirements.txt
```

### 步骤9: 下载LLM模型（可选）

```bash
# 安装Hugging Face CLI
pip install huggingface-hub

# 下载模型
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir models/llama-3.2-1b
```

---

## Docker部署

### 前置要求

- Docker 20.10+
- Docker Compose 1.29+

### 步骤1: 安装Docker

**Windows:**
1. 下载Docker Desktop: https://www.docker.com/products/docker-desktop
2. 安装并启动

**Linux:**
```bash
# 安装Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 验证
docker --version
docker-compose --version
```

### 步骤2: 构建镜像

```bash
# 构建Docker镜像
docker-compose build

# 或使用预构建镜像
docker pull your-repo/llm-rl-mt5-trading:latest
```

### 步骤3: 配置环境变量

创建 `.env` 文件：
```bash
# MT5配置
MT5_SERVER=YourBroker-Server
MT5_LOGIN=12345678
MT5_PASSWORD=your_password

# Redis配置
REDIS_HOST=redis
REDIS_PORT=6379

# 其他配置
DEBUG_MODE=false
SIMULATION_MODE=true
```

### 步骤4: 启动服务

```bash
# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### Docker Compose配置

```yaml
version: '3.8'

services:
  trading-system:
    build: .
    container_name: llm-rl-trading
    volumes:
      - ./config:/app/config
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
    environment:
      - MT5_SERVER=${MT5_SERVER}
      - MT5_LOGIN=${MT5_LOGIN}
      - MT5_PASSWORD=${MT5_PASSWORD}
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: trading-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

volumes:
  redis-data:
```

---

## 配置步骤

### 1. 基础配置

复制配置模板：
```bash
cp config/config.yaml.example config/config.yaml
```

编辑 `config/config.yaml`：

```yaml
# MT5连接
mt5:
  server: "YourBroker-Server"
  login: 12345678
  password: "your_password"
  path: "C:\\Program Files\\MetaTrader 5\\terminal64.exe"  # Windows
  # path: "/home/user/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe"  # Linux

# 数据库
database:
  sqlite_path: "data/trading.db"
  redis_host: "localhost"
  redis_port: 6379

# 风险管理
risk:
  max_risk_per_trade: 0.02
  max_daily_drawdown: 0.05
  max_positions: 10

# 交易配置
trading:
  symbols:
    - EURUSD
  default_lot_size: 0.01
  strategies_enabled:
    - trend

# 系统设置
simulation_mode: true  # 首次使用建议开启
debug_mode: false
```

### 2. 品种配置

创建品种配置文件 `config/symbols/eurusd.yaml`：

```yaml
symbol: EURUSD
spread_limit: 2.0
min_lot: 0.01
max_lot: 1.0
lot_step: 0.01
risk_multiplier: 1.0

strategies:
  - trend
  - scalp

timeframes:
  - M5
  - M15
  - H1

trading_hours:
  monday: "00:00-23:59"
  tuesday: "00:00-23:59"
  wednesday: "00:00-23:59"
  thursday: "00:00-23:59"
  friday: "00:00-22:00"
```

### 3. EA31337配置

1. 复制EA文件到MT5：
```bash
# Windows
copy ea31337\EA31337.ex5 "%APPDATA%\MetaQuotes\Terminal\<TERMINAL_ID>\MQL5\Experts\"

# Linux (Wine)
cp ea31337/EA31337.ex5 ~/.wine/drive_c/users/$USER/AppData/Roaming/MetaQuotes/Terminal/<TERMINAL_ID>/MQL5/Experts/
```

2. 在MT5中加载EA：
   - 打开MT5
   - 导航器 → Expert Advisors → EA31337
   - 拖拽到图表
   - 设置参数

3. 配置通信路径：
   - EA设置中指定 `ea31337/` 目录
   - 确保EA有读写权限

### 4. 告警配置

创建 `config/alert_config.yaml`：

```yaml
telegram:
  enabled: true
  bot_token: "your_bot_token"
  chat_id: "your_chat_id"

email:
  enabled: false
  smtp_server: "smtp.gmail.com"
  smtp_port: 587
  from_email: "your_email@gmail.com"
  password: "your_app_password"
  to_email: "your_email@gmail.com"

triggers:
  - type: "daily_loss"
    threshold: 0.03
    severity: "high"
  
  - type: "margin_level"
    threshold: 200
    severity: "critical"
```

---

## 验证安装

### 运行验证脚本

```bash
python validate_setup.py
```

**预期输出：**
```
=== 系统安装验证 ===

✓ Python版本: 3.9.x
✓ 所有Python依赖已安装
✓ MT5已安装
✓ MT5路径正确
✓ Redis服务运行中
✓ 目录结构正确
✓ 配置文件有效
⚠ LLM模型未找到（可选功能）

安装验证完成！
系统已准备就绪。

建议:
1. 下载LLM模型以启用智能分析功能
2. 在模拟模式下测试系统
3. 查看用户手册了解使用方法
```

### 手动验证

#### 1. 验证Python环境

```python
# test_python.py
import sys
print(f"Python版本: {sys.version}")

# 测试关键库
try:
    import MetaTrader5 as mt5
    print("✓ MetaTrader5库")
except ImportError:
    print("✗ MetaTrader5库未安装")

try:
    import pandas as pd
    print("✓ pandas")
except ImportError:
    print("✗ pandas未安装")

try:
    import redis
    print("✓ redis")
except ImportError:
    print("✗ redis未安装")

try:
    import talib
    print("✓ ta-lib")
except ImportError:
    print("✗ ta-lib未安装")
```

运行：
```bash
python test_python.py
```

#### 2. 验证MT5连接

```python
# test_mt5.py
import MetaTrader5 as mt5

# 初始化MT5
if not mt5.initialize():
    print(f"✗ MT5初始化失败: {mt5.last_error()}")
    exit()

print("✓ MT5初始化成功")

# 获取账户信息
account_info = mt5.account_info()
if account_info:
    print(f"✓ 账户连接成功")
    print(f"  账户: {account_info.login}")
    print(f"  余额: {account_info.balance}")
    print(f"  服务器: {account_info.server}")
else:
    print("✗ 无法获取账户信息")

mt5.shutdown()
```

运行：
```bash
python test_mt5.py
```

#### 3. 验证Redis连接

```python
# test_redis.py
import redis

try:
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.ping()
    print("✓ Redis连接成功")
    
    # 测试读写
    r.set('test_key', 'test_value')
    value = r.get('test_key')
    if value == b'test_value':
        print("✓ Redis读写正常")
    r.delete('test_key')
    
except Exception as e:
    print(f"✗ Redis连接失败: {e}")
```

运行：
```bash
python test_redis.py
```

#### 4. 验证配置文件

```bash
python -c "from src.core.config import load_config; config = load_config(); print('✓ 配置有效' if config.validate() else '✗ 配置无效')"
```

---

## 常见安装问题

### 问题1: ta-lib安装失败

**Windows:**
```bash
# 下载预编译版本
# 访问: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# 下载对应Python版本的.whl文件
pip install TA_Lib-0.4.24-cp39-cp39-win_amd64.whl
```

**Linux:**
```bash
# 安装依赖
sudo apt install libta-lib0-dev
pip install ta-lib
```

### 问题2: MetaTrader5库安装失败

```bash
# 确保使用64位Python
python -c "import struct; print(struct.calcsize('P') * 8)"
# 应输出: 64

# 重新安装
pip uninstall MetaTrader5
pip install MetaTrader5
```

### 问题3: Redis无法启动

**Windows:**
```bash
# 检查服务
sc query Redis

# 启动服务
net start Redis

# 或使用Memurai
net start Memurai
```

**Linux:**
```bash
# 检查状态
sudo systemctl status redis-server

# 启动服务
sudo systemctl start redis-server

# 查看日志
sudo journalctl -u redis-server
```

### 问题4: 权限问题

**Windows:**
```bash
# 以管理员身份运行命令提示符
# 右键 → 以管理员身份运行
```

**Linux:**
```bash
# 添加用户到必要的组
sudo usermod -aG docker $USER

# 设置目录权限
chmod -R 755 llm-rl-mt5-trading/
```

---

## 下一步

安装完成后：

1. 阅读[用户手册](user_manual.md)
2. 查看[配置指南](user_manual.md#配置指南)
3. 运行第一个策略
4. 加入社区讨论

---

## 获取帮助

如遇到安装问题：

1. 查看[故障排除指南](troubleshooting_guide.md)
2. 运行诊断脚本: `python diagnose.py`
3. 提交GitHub Issue
4. 联系技术支持

祝安装顺利！
