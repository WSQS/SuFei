# Release 发布指南

本仓库使用 GitHub Actions 自动化构建并发布 Release APK。CI 在每次 push 到 `dev` 或提交 PR 时跑构建校验；Release 在 push `v*` 标签时自动打签名包并发布到 GitHub Releases。

## 前置准备（一次性）

### 1. 生成签名 keystore

在本机执行（需要 JDK 的 `keytool`）：

```bash
keytool -genkeypair -v \
  -keystore sufei-release.jks \
  -keyalg RSA -keysize 2048 -validity 10000 \
  -alias sufei
```

按提示设置 keystore 密码、key 密码、姓名等。**生成的 `.jks` 文件务必妥善保管，丢失后将无法给后续版本签名升级。**

### 2. 上传 Secrets 到 GitHub

进入 `Settings → Secrets and variables → Actions → New repository secret`，添加以下 4 个：

| Secret 名 | 值 |
|---|---|
| `SIGNING_KEYSTORE` | keystore 文件的 base64 编码（见下方命令） |
| `KEYSTORE_PASSWORD` | keystore 密码 |
| `KEY_ALIAS` | key 别名（如 `sufei`） |
| `KEY_PASSWORD` | key 密码 |

base64 编码命令：

**macOS / Linux**：
```bash
base64 -i sufei-release.jks | pbcopy    # macOS
base64 -w 0 sufei-release.jks           # Linux
```

**Windows PowerShell**：
```powershell
$base64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("sufei-release.jks"))
Set-Content -Path "keystore.b64" -Value $base64 -NoNewline
```
将生成的字符串复制到 GitHub Secret。

### 3. 本机构建（可选）

如需本机构建签名 Release，在项目根目录创建 `keystore.properties`（已被 gitignore）：

```properties
storeFile=/absolute/path/to/sufei-release.jks
storePassword=你的keystore密码
keyAlias=sufei
keyPassword=你的key密码
```

然后：
```bash
./gradlew assembleRelease
```

## 发布流程

1. 更新 `app/build.gradle.kts` 中的 `versionCode` 和 `versionName`，提交到 `dev`
2. 打标签并推送：
   ```bash
   git tag v1.6.1
   git push origin v1.6.1
   ```
3. GitHub Actions 自动触发：构建签名 Release APK → 创建 GitHub Release → 附带 APK 下载

## 流水线说明

| Workflow | 文件 | 触发 | 作用 |
|---|---|---|---|
| CI | `.github/workflows/ci.yml` | push 到 `dev` / PR 到 `dev` | 跑 `assembleDebug` + 单元测试，校验代码可编译 |
| Release | `.github/workflows/release.yml` | push `v*` 标签 | 解密 keystore → `assembleRelease` → 发布 GitHub Release |

## fork 说明

- `app/build.gradle.kts` 的条件签名块是 fork-specific 改动，在无 `keystore.properties` 时静默跳过，不影响上游 reproducible build 约定。
- 上游原始 `build.gradle.kts` 不含 signingConfig，合并时请保留此条件块。
