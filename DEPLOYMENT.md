# 项目部署指南

## 🚀 快速启动

### 前端应用

```bash
./start-frontend.sh
```
应用将在 `http://localhost:3000` 启动。

### 后端服务 (Neo4j)

使用 Docker 快速启动 Neo4j 实例：
```bash
docker run --name neo4j-graph-template \
    -p 7474:7474 -p 7687:7687 \
    -d \
    -e NEO4J_AUTH=neo4j/your_password \
    neo4j:latest
```

## 🎨 定制与配置

### 1. 应用配置

所有前端的可配置项都集中在 `frontend/src/config.ts` 文件中。您可以修改此文件来定制：

- **应用标题与副标题**
- **社交与赞助二维码信息**

### 2. 图谱样式与布局

图谱的视觉表现（如节点、边的样式和布局算法）可以在 `frontend/src/constants/graph.ts` 中定义。这使您可以轻松地为不同类型的实体和关系设置独特的视觉风格。

### 3. 模拟数据

前端的默认数据由 `frontend/src/utils/mockData.ts` 提供。您可以修改此文件来测试不同的数据结构或在没有后端连接的情况下进行开发。

## 🔧 数据处理流程

1.  **知识抽取**:
    - 将源文本文件放入 `data/` 目录。
    - 在 `scripts/extract_kg.py` 中定制抽取逻辑（例如，修改 Prompt）。
    - 运行脚本生成知识图谱数据 (`kg_data.json`)。

2.  **数据导入**:
    - 确保 Neo4j 服务正在运行。
    - 运行 `scripts/import_to_neo4j.py` 将 `kg_data.json` 中的数据导入数据库。

## ✅ 功能特性

- **现代化前端架构**: React 18, TypeScript, Vite 6
- **强大的图可视化引擎**: AntV G6 5.0
- **企业级UI组件库**: Ant Design 5
- **模块化组件设计**
- **响应式UI设计**
- **代码规范与Linting配置**

## 📞 联系方式

- **开发者**: kuhung
- **邮箱**: hi@kuhung.me
- **项目年份**: 2025年
