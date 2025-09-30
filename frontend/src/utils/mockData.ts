import { GraphData, NodeData, EdgeData } from '@/types/graph'
import { NODE_TYPE_CONFIGS, EDGE_TYPE_CONFIGS } from '@/constants/graph'

// 示例实体类型
const ENTITY_TYPES = {
  PERSON: 'person',
  ORGANIZATION: 'organization',
  LOCATION: 'location',
  ITEM: 'item',
};

// 示例关系类型
const RELATIONSHIP_TYPES = {
  BELONGS_TO: 'belongs_to',
  LOCATED_AT: 'located_at',
  OWNS: 'owns',
  FRIEND: 'friend',
};

// 通用实体数据
const ENTITIES = [
  { id: 'ent1', label: '实体 A', type: ENTITY_TYPES.PERSON, desc: '这是一个人物实体' },
  { id: 'ent2', label: '实体 B', type: ENTITY_TYPES.PERSON, desc: '这是另一个人玝实体' },
  { id: 'org1', label: '组织 X', type: ENTITY_TYPES.ORGANIZATION, desc: '这是一个组织' },
  { id: 'loc1', label: '地点 Y', type: ENTITY_TYPES.LOCATION, desc: '这是一个地点' },
  { id: 'item1', label: '物品 Z', type: ENTITY_TYPES.ITEM, desc: '这是一个物品' }
];

// 生成关系边
function generateEdges(): EdgeData[] {
  return [
    { id: 'e1', source: 'ent1', target: 'ent2', edgeType: RELATIONSHIP_TYPES.FRIEND, label: '朋友' },
    { id: 'e2', source: 'ent1', target: 'org1', edgeType: RELATIONSHIP_TYPES.BELONGS_TO, label: '属于' },
    { id: 'e3', source: 'org1', target: 'loc1', edgeType: RELATIONSHIP_TYPES.LOCATED_AT, label: '位于' },
    { id: 'e4', source: 'ent2', target: 'item1', edgeType: RELATIONSHIP_TYPES.OWNS, label: '拥有' }
  ];
}

// 生成所有节点数据
function generateNodes(): NodeData[] {
  return ENTITIES.map(entity => ({
    id: entity.id,
    label: entity.label,
    nodeType: entity.type,
    properties: {
      description: entity.desc,
      category: entity.type
    },
    color: NODE_TYPE_CONFIGS[entity.type as keyof typeof NODE_TYPE_CONFIGS]?.color || '#5B8FF9',
    size: NODE_TYPE_CONFIGS[entity.type as keyof typeof NODE_TYPE_CONFIGS]?.size || 30
  }));
}

// 生成演示数据
export function generateDemoData(): GraphData {
  const nodes = generateNodes()
  const edges = generateEdges().map(edge => ({
    ...edge,
    color: EDGE_TYPE_CONFIGS[edge.edgeType as keyof typeof EDGE_TYPE_CONFIGS]?.color || '#e2e2e2'
  }))

  return {
    nodes,
    edges
  }
}

// 根据类型过滤数据
export function filterDataByType(data: GraphData, nodeTypes: string[] = [], edgeTypes: string[] = []): GraphData {
  let filteredNodes = data.nodes
  let filteredEdges = data.edges

  if (nodeTypes.length > 0) {
    filteredNodes = data.nodes.filter(node => nodeTypes.includes(node.nodeType))
  }

  if (edgeTypes.length > 0) {
    filteredEdges = data.edges.filter(edge => edgeTypes.includes(edge.edgeType))
  }

  // 确保边的源节点和目标节点都在过滤后的节点中
  const nodeIds = new Set(filteredNodes.map(node => node.id))
  filteredEdges = filteredEdges.filter(edge => 
    nodeIds.has(edge.source) && nodeIds.has(edge.target)
  )

  return {
    nodes: filteredNodes,
    edges: filteredEdges
  }
}

// 根据关键词搜索节点
export function searchNodes(data: GraphData, keyword: string): NodeData[] {
  if (!keyword.trim()) {
    return data.nodes
  }

  const lowerKeyword = keyword.toLowerCase()
  return data.nodes.filter(node => 
    node.label.toLowerCase().includes(lowerKeyword) ||
    node.properties?.description?.toLowerCase().includes(lowerKeyword)
  )
}
