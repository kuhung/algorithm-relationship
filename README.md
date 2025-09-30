# 知识图谱可视化模板

本项目是一个现代化的知识图谱可视化应用模板，旨在帮助开发者快速构建、部署和定制自己的图谱应用。

## 核心特性

- **现代化技术栈**: React + TypeScript + AntV G6，提供高效、可靠的开发体验。
- **模块化设计**: 清晰的项目结构，易于扩展和维护。
- **配置驱动**: 通过简单的配置即可定制应用标题、数据源、样式等。
- **端到端解决方案**: 包含从知识抽取、数据存储到前端可视化的完整工作流。

## 技术栈

- **前端**: React, TypeScript, AntV G6, Ant Design, Vite
- **后端 (示例)**: Python, Neo4j
- **知识抽取 (示例)**: `kg-gen` (基于大语言模型)

## 快速开始

### 1. 环境准备

- Node.js >= 18.0.0
- Python 3.8+
- Docker

### 2. 启动前端应用

```bash
cd frontend
npm install
npm run dev
```
应用将在 `http://localhost:3000` 启动。

### 3. 准备后端服务

#### a. 启动 Neo4j 数据库

```bash
docker run --name neo4j-graph-template \
    -p 7474:7474 -p 7687:7687 \
    -d \
    -e NEO4J_AUTH=neo4j/your_password \
    neo4j:latest
```

#### b. 准备数据和脚本

1. 将你的文本数据放入 `data/` 目录。
2. 配置 Python 虚拟环境并安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. (可选) 运行知识抽取脚本：
   ```bash
   # 需要设置 OPENAI_API_KEY 环境变量
   python scripts/extract_kg.py
   ```
4. 将抽取出的 `data/kg_data.json` 导入 Neo4j:
   ```bash
   python scripts/import_to_neo4j.py
   ```

## 个性化指南

### 1. 定制前端

- **应用信息**: 修改 `frontend/src/config.ts` 文件，可以更改应用标题、副标题和社交二维码等信息。
- **示例数据**: `frontend/src/utils/mockData.ts` 提供了前端的模拟数据结构，你可以根据自己的需求进行修改。
- **图谱样式**: 在 `frontend/src/constants/graph.ts` 中，你可以定义新的节点、边的类型、颜色、大小和布局配置。

### 2. 扩展知识抽取

- **修改 Prompt**: 在 `scripts/extract_kg.py` 中，你可以修改 `context_prompt` 来指导大语言模型抽取出你需要的实体和关系。
- **更换模型**: `kg-gen` 支持多种大语言模型，详情请参考其官方文档。

## 目录结构

```
.
├── data/                         # 数据目录 (用于存放原始数据和抽取结果)
├── scripts/                      # 脚本目录 (知识抽取、数据导入等)
├── frontend/                     # 前端应用目录
│   ├── src/
│   │   ├── components/           # React组件
│   │   ├── config.ts             # 应用配置文件
│   │   ├── constants/            # 图谱常量配置
│   │   └── utils/                # 工具函数与模拟数据
│   └── package.json
└── README.md                     # 项目主说明
```

## 参与贡献

欢迎通过 Pull Request 或 Issues 为本项目做出贡献。

## 许可证

本项目采用 MIT 许可证。
