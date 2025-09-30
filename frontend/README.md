# 知识图谱前端应用模板

这是一个基于 React + TypeScript + AntV G6 的知识图谱可视化前端应用模板。

## ✨ 特性

- 🎨 **现代化UI设计**: 基于Ant Design设计语言。
- 📊 **强大的图谱可视化**: 使用AntV G6最新版本，支持多种布局算法。
- 🎛️ **丰富的交互功能**: 拖拽、缩放、节点选择等。
- 🔍 **智能搜索**: 支持节点名称和描述的模糊搜索。
- 📱 **响应式设计**: 支持桌面端和移动端。
- 🎯 **类型安全**: 全量TypeScript开发。
- 🔧 **高度可配置**: 轻松定制应用信息、图谱样式和数据。

## 🚀 快速开始

### 环境要求

- Node.js >= 18.0.0
- npm >= 9.0.0

### 安装与启动

```bash
# 进入前端目录
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

应用将在 `http://localhost:3000` 启动。

## 🔧 自定义配置

### 1. 修改应用信息

编辑 `src/config.ts` 文件，可以修改整个应用的元信息，如标题、副标题等。

```typescript
// src/config.ts
export const appConfig = {
  title: 'Your Project Title',
  subtitle: 'Your Project Subtitle',
  // ... 其他配置
};
```

### 2. 定义节点与关系类型

在 `src/constants/graph.ts` 中，您可以为不同的节点和关系类型定义样式（颜色、大小等）。

```typescript
// src/constants/graph.ts
export const NODE_TYPE_CONFIGS = {
  'person': { color: '#5B8FF9', size: 30, label: '人物' },
  // ... 添加你的类型
};

export const EDGE_TYPE_CONFIGS = {
  'friend': { color: '#e2e2e2', label: '朋友' },
  // ... 添加你的关系
};
```

### 3. 修改布局算法

同样在 `src/constants/graph.ts`，您可以配置或添加新的 G6 布局算法。

### 4. 使用你自己的数据

当前版本使用 `src/utils/mockData.ts` 中的模拟数据。如需连接真实后端API，可修改 `src/views/GraphView.tsx` 中的数据加载逻辑。

## 🏗️ 项目结构

```
frontend/
├── src/
│   ├── components/     # React组件
│   ├── views/          # 页面组件
│   ├── types/          # TypeScript类型定义
│   ├── config.ts       # 应用级配置文件
│   ├── constants/      # 图谱常量与样式配置
│   ├── utils/          # 工具函数与模拟数据
│   └── ...
├── package.json        # 项目配置
└── README.md           # 本文档
```

## 🤝 参与贡献

欢迎为此模板项目贡献代码或提出建议。

## 📄 许可证

本项目采用 MIT 许可证。
