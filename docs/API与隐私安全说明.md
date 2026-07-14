# API 与隐私安全说明

## 自动生成接口文档

- 在线 Swagger: `/docs`
- 在线 ReDoc: `/redoc`
- OpenAPI JSON: `/openapi.json`

导出接口文档：

```powershell
cd D:\code\python\car-vision-agent-system
.\.venv\Scripts\python.exe .\scripts\export_openapi.py
```

导出结果默认写入 `docs/openapi.json`。

## 敏感字段加密

当前系统对以下用户敏感字段采用字段级加密存储：

- 邮箱
- 手机号
- 微信 OpenID

实现方式：

- 密码：`bcrypt` 哈希
- 邮箱/手机号/OpenID：`AES-GCM` 加密存储
- 精确查询与唯一校验：基于密钥的 `HMAC-SHA256` 哈希

## 传输层加密

代码层已经支持：

- JWT Bearer 鉴权
- SMTP SSL 邮件发送

部署时建议统一使用 HTTPS/WSS 访问前后端接口与实时推送通道。若本地已有证书，可使用带 SSL 参数的 `uvicorn` 启动，或在 Nginx / Caddy 上终止 TLS。

## 密钥生成

生成 `.env` 所需的加密密钥：

```powershell
cd D:\code\python\car-vision-agent-system
.\.venv\Scripts\python.exe .\scripts\generate_security_keys.py
```

将输出写入 `backend/.env`：

- `DATA_ENCRYPTION_KEY`
- `DATA_HASH_KEY`

## 数据迁移

初始化数据库时会自动执行以下动作：

- 自动补充 `users` 表中的加密列与哈希列
- 自动将旧的明文邮箱、手机号、微信 OpenID 回填到加密列
- 成功迁移后清空旧明文字段

执行初始化：

```powershell
cd D:\code\python\car-vision-agent-system
.\.venv\Scripts\python.exe .\scripts\init_db.py
```
