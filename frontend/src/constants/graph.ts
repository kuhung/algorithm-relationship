// 图布局类型
export enum GraphLayoutType {
  FORCE = 'force',
  CIRCULAR = 'circular',
  RADIAL = 'radial',
  DAGRE = 'dagre',
  GRID = 'grid',
  CONCENTRIC = 'concentric'
}

// 图布局配置
export const LAYOUT_CONFIGS = {
  [GraphLayoutType.FORCE]: {
    type: 'force',
    linkDistance: 90,
    preventOverlap: false,
    nodeSize: 30,
    nodeSpacing: 20,
    centrality: 'degree',
    animate: false, // 关闭布局动画
    maxIteration: 200, // 限制最大迭代次数，防止过度计算
    enableTick: false, // 禁用持续的 tick
    onLayoutEnd: () => {
      // 布局结束后的回调
      console.log('Force layout completed')
    }
  },
  [GraphLayoutType.CIRCULAR]: {
    type: 'circular',
    radius: 200,
    startRadius: 100,
    endRadius: 300,
    animate: false // 增加animate属性
  },
  [GraphLayoutType.RADIAL]: {
    type: 'radial',
    linkDistance: 120,
    unitRadius: 80,
    preventOverlap: true,
    nodeSize: 30,
    animate: false // 增加animate属性
  },
  [GraphLayoutType.DAGRE]: {
    type: 'dagre',
    rankdir: 'TB',
    nodesep: 20,
    ranksep: 50,
    animate: false // 增加animate属性
  },
  [GraphLayoutType.GRID]: {
    type: 'grid',
    preventOverlap: true,
    nodeSize: 30,
    sortBy: 'degree',
    animate: false // 增加animate属性
  },
  [GraphLayoutType.CONCENTRIC]: {
    type: 'concentric',
    nodeSize: 30,
    minNodeSpacing: 20,
    preventOverlap: true,
    animate: false // 增加animate属性
  }
}

// 节点类型配置
export const NODE_TYPE_CONFIGS = {
  'concept': {
    color: '#5B8FF9',
    size: 35,
    label: '概念/理论'
  },
  'algorithm': {
    color: '#E86452',
    size: 30,
    label: '算法'
  },
  'data_structure': {
    color: '#5AD8A6',
    size: 28,
    label: '数据结构'
  },
  'technology': {
    color: '#F6BD16',
    size: 25,
    label: '技术/工具'
  },
  'skill': {
    color: '#845EC2',
    size: 22,
    label: '技能'
  },
  'career_level': {
    color: '#FF6B6B',
    size: 40,
    label: '职业级别'
  },
  'domain': {
    color: '#6DC8EC',
    size: 38,
    label: '专业领域'
  },
  'model': {
    color: '#9C88FF',
    size: 32,
    label: '模型/架构'
  },
  'task': {
    color: '#5D7092',
    size: 26,
    label: '任务类型'
  },
  'metric': {
    color: '#FFC75F',
    size: 20,
    label: '评估指标'
  }
}

// 边类型配置
export const EDGE_TYPE_CONFIGS = {
  'contains': {
    color: '#6DC8EC',
    label: '包含'
  },
  'is_a': {
    color: '#5B8FF9',
    label: '是'
  },
  'applies_to': {
    color: '#5AD8A6',
    label: '应用于'
  },
  'implements': {
    color: '#F6BD16',
    label: '实现'
  },
  'depends_on': {
    color: '#E86452',
    label: '依赖于'
  },
  'affects': {
    color: '#845EC2',
    label: '影响'
  },
  'compares_with': {
    color: '#9C88FF',
    label: '对比'
  },
  'evaluates': {
    color: '#FFC75F',
    label: '评估'
  }
}

// 默认节点样式
export const DEFAULT_NODE_STYLE = {
  size: 30,
  labelCfg: {
    position: 'bottom',
    style: {
      fontSize: 12,
      fill: '#333',
      fontWeight: 500,
      background: {
        fill: 'rgba(255, 255, 255, 0.8)',
        padding: [2, 4, 2, 4],
        radius: 4
      }
    }
  },
  style: {
    fill: '#5B8FF9',
    stroke: '#fff',
    lineWidth: 2,
    cursor: 'pointer'
  },
  stateStyles: {
    hover: {
      lineWidth: 3,
      shadowColor: '#000',
      shadowBlur: 10,
      shadowOffsetX: 0,
      shadowOffsetY: 0
    },
    selected: {
      lineWidth: 4,
      stroke: '#1890ff',
      shadowColor: '#1890ff',
      shadowBlur: 15,
      shadowOffsetX: 0,
      shadowOffsetY: 0
    }
  }
}

// 默认边样式
export const DEFAULT_EDGE_STYLE = {
  style: {
    stroke: '#e2e2e2',
    lineWidth: 2,
    cursor: 'pointer'
  },
  labelCfg: {
    style: {
      fontSize: 10,
      fill: '#666',
      background: {
        fill: 'rgba(255, 255, 255, 0.9)',
        padding: [2, 4, 2, 4],
        radius: 2
      }
    }
  },
  stateStyles: {
    hover: {
      lineWidth: 3,
      stroke: '#1890ff'
    },
    selected: {
      lineWidth: 4,
      stroke: '#1890ff'
    }
  }
}

// 图谱主题配置
export const GRAPH_THEMES = {
  default: {
    background: '#fafafa',
    nodeColors: ['#5B8FF9', '#5AD8A6', '#5D7092', '#F6BD16', '#E86452', '#6DC8EC'],
    edgeColor: '#e2e2e2'
  },
  dark: {
    background: '#1f1f1f',
    nodeColors: ['#4A90E2', '#7ED321', '#9013FE', '#F5A623', '#D0021B', '#50E3C2'],
    edgeColor: '#666'
  }
}

// 默认选中节点类型
export const DEFAULT_SELECTED_NODE_TYPES = ['concept', 'domain', 'career_level']
