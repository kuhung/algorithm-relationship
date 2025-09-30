import { useEffect, useState, useMemo } from 'react'
import { Layout, Spin, App as AntdApp, FloatButton } from 'antd'
import { SettingOutlined, ProfileOutlined } from '@ant-design/icons'
import GraphContainer from '@/components/GraphContainer'
import GraphToolbar from '@/components/GraphToolbar'
import GraphSidebar from '@/components/GraphSidebar'
import BrandLogo from '@/components/BrandLogo'
import QRCodeModal from '@/components/QRCodeModal'
import { GraphData, GraphConfig, NodeData } from '@/types/graph'
import { generateDemoData } from '@/utils/Data'
import { GraphLayoutType, NODE_TYPE_CONFIGS, DEFAULT_SELECTED_NODE_TYPES } from '@/constants/graph'
import useResponsive from '@/utils/useResponsive'
import { appConfig } from '@/config'

const { Content } = Layout

interface GraphViewState {
  data: GraphData
  loading: boolean
  selectedNode: NodeData | null
  config: GraphConfig
}

interface ModalState {
  sponsor: boolean
  joinGroup: boolean
}

const GraphView = () => {
  const { message } = AntdApp.useApp() // Use App.useApp() to get message instance
  const { isMobile } = useResponsive()

  const [state, setState] = useState<GraphViewState>({
    data: { nodes: [], edges: [] },
    loading: true,
    selectedNode: null,
    config: {
      layout: GraphLayoutType.FORCE,
      nodeSize: 30,
      showNodeLabel: true,
      showEdgeLabel: false,
      enableAnimation: true,
    }
  })
  const allNodeTypes = useMemo(() => Object.keys(NODE_TYPE_CONFIGS), [])
  const initialHiddenNodeTypes = useMemo(() => allNodeTypes.filter(type => !DEFAULT_SELECTED_NODE_TYPES.includes(type)), [allNodeTypes])
  const [hiddenNodeTypes, setHiddenNodeTypes] = useState<string[]>(initialHiddenNodeTypes)
  const [modalState, setModalState] = useState<ModalState>({
    sponsor: false,
    joinGroup: false
  })
  const [toolbarVisible, setToolbarVisible] = useState(false)
  const [sidebarVisible, setSidebarVisible] = useState(false)


  // 初始化数据
  useEffect(() => {
    const loadData = async () => {
      try {
        setState(prev => ({ ...prev, loading: true }))
        
        // 模拟加载时间
        await new Promise(resolve => setTimeout(resolve, 1000))
        
        // 生成演示数据
        const demoData = generateDemoData()
        
        setState(prev => ({
          ...prev,
          data: demoData,
          loading: false
        }))
        
        // 移除旧的初始化逻辑，因为 hiddenNodeTypes 已在 useState 中初始化
        // const allNodeTypes = Object.keys(NODE_TYPE_CONFIGS)
        // const initialHiddenNodeTypes = allNodeTypes.filter(type => !DEFAULT_SELECTED_NODE_TYPES.includes(type))
        // setHiddenNodeTypes(initialHiddenNodeTypes)

        message.success('数据加载完成')
      } catch (error) {
        console.error('数据加载失败:', error)
        message.error('数据加载失败，请稍后重试')
        setState(prev => ({ ...prev, loading: false }))
      }
    }

    loadData()
  }, [])

  // 处理节点选择
  const handleNodeSelect = (node: NodeData | null) => {
    setState(prev => ({ ...prev, selectedNode: node }))
  }

  // 处理布局变更
  const handleLayoutChange = (layout: GraphLayoutType) => {
    setState(prev => ({
      ...prev,
      config: { ...prev.config, layout }
    }))
  }

  // 处理配置变更
  const handleConfigChange = (newConfig: Partial<GraphConfig>) => {
    setState(prev => ({
      ...prev,
      config: { ...prev.config, ...newConfig }
    }))
  }

  // 打开/关闭模态框
  const handleShowSponsorModal = () => setModalState(prev => ({ ...prev, sponsor: true }))
  const handleCloseSponsorModal = () => setModalState(prev => ({ ...prev, sponsor: false }))
  const handleShowJoinGroupModal = () => setModalState(prev => ({ ...prev, joinGroup: true }))
  const handleCloseJoinGroupModal = () => setModalState(prev => ({ ...prev, joinGroup: false }))

  // 处理数据刷新
  const handleDataRefresh = () => {
    setState(prev => ({
      ...prev,
      data: generateDemoData(),
      selectedNode: null
    }))
    message.success('数据已刷新')
  }

  // 处理节点类型选择
  const handleNodeTypeSelect = (selectedNodeTypes: string[]) => {
    const newHiddenNodeTypes = allNodeTypes.filter(type => !selectedNodeTypes.includes(type));
    setHiddenNodeTypes(newHiddenNodeTypes)
  }

  // 根据隐藏节点类型过滤数据
  const filteredGraphData = useMemo(() => {
    const visibleNodes = state.data.nodes.filter(node => !hiddenNodeTypes.includes(node.nodeType))
    const visibleNodeIds = new Set(visibleNodes.map(node => node.id))
    const visibleEdges = state.data.edges.filter(edge =>
      visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target)
    )
    return {
      nodes: visibleNodes,
      edges: visibleEdges
    }
  }, [state.data, hiddenNodeTypes])

  // 计算统计数据
  const stats = useMemo(() => ({
    nodeCount: state.data.nodes.length,
    edgeCount: state.data.edges.length,
    nodeTypes: [...new Set(state.data.nodes.map(n => n.nodeType))].length,
    edgeTypes: [...new Set(state.data.edges.map(e => e.edgeType))].length
  }), [state.data])

  if (state.loading) {
    return (
      <div className="loading-container">
        <Spin size="large" />
        <div className="loading-title">{appConfig.title}</div>
        <div className="loading-subtitle">正在加载知识图谱...</div>
      </div>
    )
  }

  return (
    <Layout style={{ height: '100vh' }}>
      <Content style={{ position: 'relative', overflow: 'hidden' }}>
        {/* 品牌标识 */}
        <BrandLogo />
        {/* 图谱容器 */}
        <GraphContainer
          data={filteredGraphData}
          config={state.config}
          onNodeSelect={handleNodeSelect}
        />
        
        {isMobile ? (
          <>
            <FloatButton.Group trigger="click" icon={<SettingOutlined />}>
              <FloatButton
                icon={<ProfileOutlined />}
                onClick={() => setSidebarVisible(true)}
              />
              <FloatButton
                icon={<SettingOutlined />}
                onClick={() => setToolbarVisible(true)}
              />
            </FloatButton.Group>

            <GraphToolbar
              config={state.config}
              onLayoutChange={handleLayoutChange}
              onConfigChange={handleConfigChange}
              onDataRefresh={handleDataRefresh}
              onShowSponsorModal={handleShowSponsorModal}
              onShowJoinGroupModal={handleShowJoinGroupModal}
              isMobile={isMobile}
              open={toolbarVisible}
              onClose={() => setToolbarVisible(false)}
            />
            
            <GraphSidebar
              selectedNode={state.selectedNode}
              stats={stats}
              data={state.data}
              onNodeSelect={handleNodeSelect}
              onNodeTypeSelect={handleNodeTypeSelect}
              isMobile={isMobile}
              open={sidebarVisible}
              onClose={() => setSidebarVisible(false)}
            />
          </>
        ) : (
          <>
            <GraphToolbar
              config={state.config}
              onLayoutChange={handleLayoutChange}
              onConfigChange={handleConfigChange}
              onDataRefresh={handleDataRefresh}
              onShowSponsorModal={handleShowSponsorModal}
              onShowJoinGroupModal={handleShowJoinGroupModal}
            />
            
            <GraphSidebar
              selectedNode={state.selectedNode}
              stats={stats}
              data={state.data}
              onNodeSelect={handleNodeSelect}
              onNodeTypeSelect={handleNodeTypeSelect}
            />
          </>
        )}
        
        {/* 二维码弹窗 */}
        <QRCodeModal
          title={appConfig.qrCodeModal.sponsorTitle}
          imageUrl={appConfig.qrCodeModal.sponsorImageUrl}
          modalText={appConfig.qrCodeModal.modalText}
          visible={modalState.sponsor}
          onClose={handleCloseSponsorModal}
        />
        <QRCodeModal
          title={appConfig.qrCodeModal.joinGroupTitle}
          imageUrl={appConfig.qrCodeModal.joinGroupImageUrl}
          modalText={appConfig.qrCodeModal.modalText}
          visible={modalState.joinGroup}
          onClose={handleCloseJoinGroupModal}
        />
      </Content>
    </Layout>
  )
}

export default GraphView
