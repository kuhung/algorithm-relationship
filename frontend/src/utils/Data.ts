import { GraphData, NodeData, EdgeData } from '@/types/graph'
import { NODE_TYPE_CONFIGS, EDGE_TYPE_CONFIGS } from '@/constants/graph'

// 算法知识图谱实体类型
const ENTITY_TYPES = {
  CONCEPT: 'concept',
  ALGORITHM: 'algorithm',
  DATA_STRUCTURE: 'data_structure',
  TECHNOLOGY: 'technology',
  SKILL: 'skill',
  CAREER_LEVEL: 'career_level',
  DOMAIN: 'domain',
  MODEL: 'model',
  TASK: 'task',
  METRIC: 'metric',
};

// 算法知识图谱关系类型
const RELATIONSHIP_TYPES = {
  /** A 包含 B (e.g., 机器学习包含监督学习) */
  CONTAINS: 'contains',
  /** A 是 B 的一种 (e.g., 冒泡排序是一种比较排序) */
  IS_A: 'is_a',
  /** A 应用于 B (e.g., CNN 应用于计算机视觉) */
  APPLIES_TO: 'applies_to',
  /** A 实现 B (e.g., Python 实现机器学习) */
  IMPLEMENTS: 'implements',
  /** A 依赖/需要 B (e.g., 机器学习依赖数学) */
  DEPENDS_ON: 'depends_on',
  /** A 影响 B (可以是正向、负向或中性影响) */
  AFFECTS: 'affects',
  /** A 与 B 对比 */
  COMPARES_WITH: 'compares_with',
  /** A 度量/评估 B */
  EVALUATES: 'evaluates',
};

// 算法知识图谱实体数据
const ENTITIES = [
  // ===== 核心概念 =====
  { id: 'ml', label: '机器学习', type: ENTITY_TYPES.CONCEPT, desc: '通过算法让计算机从数据中学习的技术' },
  { id: 'dsa', label: '数据结构与算法', type: ENTITY_TYPES.CONCEPT, desc: '高效组织、处理和存储数据的基本工具集' },
  { id: 'computer_science', label: '计算机科学', type: ENTITY_TYPES.CONCEPT, desc: '算法工程师知识体系的基础学科' },
  { id: 'mathematics', label: '数学基础', type: ENTITY_TYPES.CONCEPT, desc: '算法工程师必备的数学知识' },
  { id: 'supervised_learning', label: '监督学习', type: ENTITY_TYPES.CONCEPT, desc: '使用标记数据训练模型的学习方式' },
  { id: 'unsupervised_learning', label: '无监督学习', type: ENTITY_TYPES.CONCEPT, desc: '从无标记数据中发现模式的学习方式' },
  { id: 'reinforcement_learning', label: '强化学习', type: ENTITY_TYPES.CONCEPT, desc: '通过与环境交互学习最优策略的学习方式' },
  { id: 'deep_learning', label: '深度学习', type: ENTITY_TYPES.CONCEPT, desc: '使用多层神经网络的机器学习方法' },
  
  // 强化学习核心概念
  { id: 'agent', label: '智能体', type: ENTITY_TYPES.CONCEPT, desc: '在强化学习中做出决策的主体' },
  { id: 'environment', label: '环境', type: ENTITY_TYPES.CONCEPT, desc: '智能体交互的外部系统' },
  { id: 'action', label: '行动', type: ENTITY_TYPES.CONCEPT, desc: '智能体可以采取的操作' },
  { id: 'reward', label: '奖励', type: ENTITY_TYPES.CONCEPT, desc: '环境对智能体行动的正面反馈' },
  { id: 'penalty', label: '惩罚', type: ENTITY_TYPES.CONCEPT, desc: '环境对智能体行动的负面反馈' },
  { id: 'policy', label: '策略', type: ENTITY_TYPES.CONCEPT, desc: '智能体在不同状态下选择行动的规则' },
  { id: 'state', label: '状态', type: ENTITY_TYPES.CONCEPT, desc: '环境的当前情况描述' },
  
  // 算法范式
  { id: 'greedy_algorithm', label: '贪心算法', type: ENTITY_TYPES.CONCEPT, desc: '每步选择当前最优解的算法策略' },
  { id: 'divide_conquer', label: '分治算法', type: ENTITY_TYPES.CONCEPT, desc: '将问题分解为子问题递归解决的策略' },
  
  // 复杂度分析
  { id: 'time_complexity', label: '时间复杂度', type: ENTITY_TYPES.CONCEPT, desc: '算法执行时间随输入规模增长的度量' },
  { id: 'space_complexity', label: '空间复杂度', type: ENTITY_TYPES.CONCEPT, desc: '算法所需存储空间随输入规模增长的度量' },
  { id: 'big_o_notation', label: '大O表示法', type: ENTITY_TYPES.CONCEPT, desc: '描述算法复杂度上界的数学记号' },
  { id: 'complexity_analysis', label: '复杂度分析', type: ENTITY_TYPES.CONCEPT, desc: '评估算法性能的分析方法' },
  { id: 'best_case', label: '最好情况', type: ENTITY_TYPES.CONCEPT, desc: '算法在最优输入下的性能表现' },
  { id: 'average_case', label: '平均情况', type: ENTITY_TYPES.CONCEPT, desc: '算法在典型输入下的性能表现' },
  { id: 'worst_case', label: '最坏情况', type: ENTITY_TYPES.CONCEPT, desc: '算法在最差输入下的性能表现' },
  
  // 机器学习核心概念
  { id: 'feature_engineering', label: '特征工程', type: ENTITY_TYPES.CONCEPT, desc: '从原始数据中提取和构建有用特征的过程' },
  { id: 'data_preprocessing', label: '数据预处理', type: ENTITY_TYPES.CONCEPT, desc: '清洗和准备原始数据用于模型训练' },
  { id: 'hyperparameter_tuning', label: '超参数调优', type: ENTITY_TYPES.CONCEPT, desc: '优化模型超参数以提升性能的过程' },
  { id: 'overfitting', label: '过拟合', type: ENTITY_TYPES.CONCEPT, desc: '模型在训练数据上表现好但泛化能力差' },
  { id: 'data_leakage', label: '数据泄露', type: ENTITY_TYPES.CONCEPT, desc: '训练数据中包含未来信息导致的问题' },
  { id: 'gradient_vanishing', label: '梯度消失', type: ENTITY_TYPES.CONCEPT, desc: '深层网络中梯度逐层衰减的问题' },
  { id: 'gradient_exploding', label: '梯度爆炸', type: ENTITY_TYPES.CONCEPT, desc: '深层网络中梯度逐层放大的问题' },
  { id: 'generalization', label: '泛化能力', type: ENTITY_TYPES.CONCEPT, desc: '模型对未见数据的预测能力' },
  { id: 'model_lifecycle', label: '模型生命周期', type: ENTITY_TYPES.CONCEPT, desc: '从问题定义到模型部署的完整流程' },
  { id: 'problem_definition', label: '问题定义', type: ENTITY_TYPES.CONCEPT, desc: '明确机器学习要解决的具体问题' },
  { id: 'model_training', label: '模型训练', type: ENTITY_TYPES.CONCEPT, desc: '使用训练数据学习模型参数的过程' },
  { id: 'model_evaluation', label: '模型评估', type: ENTITY_TYPES.CONCEPT, desc: '评估模型性能和泛化能力的过程' },
  { id: 'optimal_substructure', label: '最优子结构', type: ENTITY_TYPES.CONCEPT, desc: '动态规划问题的基本性质' },
  { id: 'overlapping_subproblems', label: '重叠子问题', type: ENTITY_TYPES.CONCEPT, desc: '动态规划中子问题重复出现的特性' },
  { id: 'greedy_choice_property', label: '贪心选择性质', type: ENTITY_TYPES.CONCEPT, desc: '贪心算法适用问题的基本特征' },
  
  // MLOps概念
  { id: 'mlops', label: 'MLOps', type: ENTITY_TYPES.CONCEPT, desc: '将DevOps实践应用于机器学习生命周期' },
  { id: 'model_deployment', label: '模型部署', type: ENTITY_TYPES.CONCEPT, desc: '将训练好的模型投入生产环境使用' },
  { id: 'model_monitoring', label: '模型监控', type: ENTITY_TYPES.CONCEPT, desc: '持续监控生产环境中模型的性能' },
  { id: 'data_drift', label: '数据漂移', type: ENTITY_TYPES.CONCEPT, desc: '生产数据分布与训练数据分布的偏差' },
  { id: 'concept_drift', label: '概念漂移', type: ENTITY_TYPES.CONCEPT, desc: '目标变量与特征关系随时间变化' },
  
  // ===== 数据结构 =====
  // 线性结构
  { id: 'linear_structure', label: '线性结构', type: ENTITY_TYPES.CONCEPT, desc: '元素按线性顺序排列的数据结构' },
  { id: 'nonlinear_structure', label: '非线性结构', type: ENTITY_TYPES.CONCEPT, desc: '元素不按线性顺序排列的数据结构' },
  { id: 'array', label: '数组', type: ENTITY_TYPES.DATA_STRUCTURE, desc: '连续内存存储的线性数据结构' },
  { id: 'cache_friendly', label: '缓存友好性', type: ENTITY_TYPES.CONCEPT, desc: '数据结构对CPU缓存的友好程度' },
  { id: 'random_access', label: '随机访问', type: ENTITY_TYPES.CONCEPT, desc: '可以直接访问任意位置元素的能力' },
  { id: 'dynamic_insertion', label: '动态插入', type: ENTITY_TYPES.CONCEPT, desc: '运行时高效插入元素的能力' },
  { id: 'dynamic_deletion', label: '动态删除', type: ENTITY_TYPES.CONCEPT, desc: '运行时高效删除元素的能力' },
  { id: 'linked_list', label: '链表', type: ENTITY_TYPES.DATA_STRUCTURE, desc: '通过指针连接的动态数据结构' },
  { id: 'stack', label: '栈', type: ENTITY_TYPES.DATA_STRUCTURE, desc: '后进先出(LIFO)的线性数据结构' },
  { id: 'queue', label: '队列', type: ENTITY_TYPES.DATA_STRUCTURE, desc: '先进先出(FIFO)的线性数据结构' },
  { id: 'binary_tree', label: '二叉树', type: ENTITY_TYPES.DATA_STRUCTURE, desc: '每个节点最多有两个子节点的树结构' },
  { id: 'binary_search_tree', label: '二叉搜索树', type: ENTITY_TYPES.DATA_STRUCTURE, desc: '具有有序性质的二叉树' },
  { id: 'tree_balance', label: '树平衡', type: ENTITY_TYPES.CONCEPT, desc: '保持树高度在对数级别的性质' },
  { id: 'tree_rotation', label: '树旋转', type: ENTITY_TYPES.CONCEPT, desc: '维持平衡二叉树平衡性的操作' },
  { id: 'tree_degeneration', label: '树退化', type: ENTITY_TYPES.CONCEPT, desc: '二叉搜索树退化为链表的现象' },
  { id: 'avl_tree', label: 'AVL树', type: ENTITY_TYPES.DATA_STRUCTURE, desc: '自平衡的二叉搜索树' },
  { id: 'red_black_tree', label: '红黑树', type: ENTITY_TYPES.DATA_STRUCTURE, desc: '自平衡的二叉搜索树变种' },
  { id: 'heap', label: '堆', type: ENTITY_TYPES.DATA_STRUCTURE, desc: '满足堆属性的完全二叉树' },
  { id: 'priority_queue', label: '优先队列', type: ENTITY_TYPES.DATA_STRUCTURE, desc: '基于优先级的队列数据结构' },
  { id: 'heap_property', label: '堆属性', type: ENTITY_TYPES.CONCEPT, desc: '父节点与子节点间的大小关系约束' },
  { id: 'complete_binary_tree', label: '完全二叉树', type: ENTITY_TYPES.CONCEPT, desc: '除最后一层外都填满的二叉树' },
  { id: 'hash_table', label: '哈希表', type: ENTITY_TYPES.DATA_STRUCTURE, desc: '基于哈希函数的键值对存储结构' },
  { id: 'hash_function', label: '哈希函数', type: ENTITY_TYPES.CONCEPT, desc: '将键映射到存储位置的函数' },
  { id: 'hash_collision', label: '哈希冲突', type: ENTITY_TYPES.CONCEPT, desc: '不同键映射到相同位置的现象' },
  { id: 'collision_resolution', label: '冲突解决', type: ENTITY_TYPES.CONCEPT, desc: '处理哈希冲突的策略' },
  { id: 'graph', label: '图', type: ENTITY_TYPES.DATA_STRUCTURE, desc: '由节点和边组成的网络结构' },
  { id: 'vertices', label: '节点', type: ENTITY_TYPES.CONCEPT, desc: '图中的基本元素' },
  { id: 'edges', label: '边', type: ENTITY_TYPES.CONCEPT, desc: '连接图中节点的关系' },
  { id: 'graph_traversal', label: '图遍历', type: ENTITY_TYPES.CONCEPT, desc: '系统访问图中所有节点的过程' },
  { id: 'network_analysis', label: '网络分析', type: ENTITY_TYPES.CONCEPT, desc: '分析网络结构和关系的方法' },
  { id: 'social_network', label: '社交网络', type: ENTITY_TYPES.TASK, desc: '分析社交关系网络的应用' },
  { id: 'logistics_routing', label: '物流路径规划', type: ENTITY_TYPES.TASK, desc: '优化物流配送路径的问题' },
  { id: 'knowledge_graph', label: '知识图谱', type: ENTITY_TYPES.CONCEPT, desc: '结构化知识表示的图形化方法' },
  
  // ===== 算法 =====
  // 算法分类
  { id: 'comparison_sort', label: '比较排序', type: ENTITY_TYPES.CONCEPT, desc: '基于元素比较的排序算法' },
  { id: 'non_comparison_sort', label: '非比较排序', type: ENTITY_TYPES.CONCEPT, desc: '不基于元素比较的排序算法' },
  { id: 'stable_sort', label: '稳定排序', type: ENTITY_TYPES.CONCEPT, desc: '保持相等元素原始顺序的排序' },
  { id: 'in_place_sort', label: '原地排序', type: ENTITY_TYPES.CONCEPT, desc: '不需要额外存储空间的排序算法' },
  
  // 排序算法
  { id: 'bubble_sort', label: '冒泡排序', type: ENTITY_TYPES.ALGORITHM, desc: '通过相邻元素比较交换的简单排序算法' },
  { id: 'insertion_sort', label: '插入排序', type: ENTITY_TYPES.ALGORITHM, desc: '通过逐个插入元素的排序算法' },
  { id: 'selection_sort', label: '选择排序', type: ENTITY_TYPES.ALGORITHM, desc: '通过选择最小元素的排序算法' },
  { id: 'merge_sort', label: '归并排序', type: ENTITY_TYPES.ALGORITHM, desc: '稳定的分治排序算法' },
  { id: 'quick_sort', label: '快速排序', type: ENTITY_TYPES.ALGORITHM, desc: '基于分治思想的高效排序算法' },
  { id: 'heap_sort', label: '堆排序', type: ENTITY_TYPES.ALGORITHM, desc: '基于堆数据结构的排序算法' },
  { id: 'counting_sort', label: '计数排序', type: ENTITY_TYPES.ALGORITHM, desc: '基于计数的非比较排序算法' },
  { id: 'radix_sort', label: '基数排序', type: ENTITY_TYPES.ALGORITHM, desc: '按位数进行的非比较排序算法' },
  { id: 'bucket_sort', label: '桶排序', type: ENTITY_TYPES.ALGORITHM, desc: '将元素分配到桶中的排序算法' },
  
  // 搜索算法
  { id: 'linear_search', label: '线性搜索', type: ENTITY_TYPES.ALGORITHM, desc: '逐个检查元素的搜索算法' },
  { id: 'binary_search', label: '二分搜索', type: ENTITY_TYPES.ALGORITHM, desc: '在有序数组中快速查找的算法' },
  
  // 图算法
  { id: 'bfs', label: '广度优先搜索', type: ENTITY_TYPES.ALGORITHM, desc: '按层次遍历图的搜索算法' },
  { id: 'dfs', label: '深度优先搜索', type: ENTITY_TYPES.ALGORITHM, desc: '沿路径深入遍历图的搜索算法' },
  { id: 'dijkstra', label: 'Dijkstra算法', type: ENTITY_TYPES.ALGORITHM, desc: '计算单源最短路径的图算法' },
  { id: 'prim', label: 'Prim算法', type: ENTITY_TYPES.ALGORITHM, desc: '构建最小生成树的贪心算法' },
  { id: 'kruskal', label: 'Kruskal算法', type: ENTITY_TYPES.ALGORITHM, desc: '构建最小生成树的另一种贪心算法' },
  
  // 其他经典算法
  { id: 'dynamic_programming', label: '动态规划', type: ENTITY_TYPES.ALGORITHM, desc: '通过存储子问题解避免重复计算的算法思想' },
  { id: 'huffman_coding', label: '霍夫曼编码', type: ENTITY_TYPES.ALGORITHM, desc: '用于数据压缩的贪心算法' },
  
  // 优化算法
  { id: 'linear_programming', label: '线性规划', type: ENTITY_TYPES.ALGORITHM, desc: '在线性约束下优化线性目标函数' },
  { id: 'integer_programming', label: '整数规划', type: ENTITY_TYPES.ALGORITHM, desc: '变量为整数的线性规划问题' },
  { id: 'heuristic_algorithm', label: '启发式算法', type: ENTITY_TYPES.ALGORITHM, desc: '用于快速找到近似解的算法' },
  { id: 'metaheuristic_algorithm', label: '元启发式算法', type: ENTITY_TYPES.ALGORITHM, desc: '指导启发式算法的高层策略' },
  { id: 'simulated_annealing', label: '模拟退火', type: ENTITY_TYPES.ALGORITHM, desc: '模拟物理退火过程的优化算法' },
  { id: 'genetic_algorithm', label: '遗传算法', type: ENTITY_TYPES.ALGORITHM, desc: '模拟生物进化的优化算法' },
  { id: 'branch_bound', label: '分支定界法', type: ENTITY_TYPES.ALGORITHM, desc: '用于求解整数规划的精确算法' },
  
  // 强化学习算法
  { id: 'q_learning', label: 'Q-Learning', type: ENTITY_TYPES.ALGORITHM, desc: '基于价值函数的强化学习算法' },
  { id: 'policy_gradient', label: '策略梯度', type: ENTITY_TYPES.ALGORITHM, desc: '直接优化策略的强化学习方法' },
  { id: 'actor_critic', label: 'Actor-Critic', type: ENTITY_TYPES.ALGORITHM, desc: '结合价值函数和策略的强化学习方法' },
  
  // 搜索和优化方法
  { id: 'grid_search', label: '网格搜索', type: ENTITY_TYPES.ALGORITHM, desc: '穷举搜索超参数组合的方法' },
  { id: 'random_search', label: '随机搜索', type: ENTITY_TYPES.ALGORITHM, desc: '随机采样超参数的优化方法' },
  { id: 'bayesian_optimization', label: '贝叶斯优化', type: ENTITY_TYPES.ALGORITHM, desc: '基于贝叶斯推理的全局优化方法' },
  
  // ===== 模型架构 =====
  // 传统机器学习模型
  { id: 'linear_regression', label: '线性回归', type: ENTITY_TYPES.MODEL, desc: '预测连续值的基础回归模型' },
  { id: 'logistic_regression', label: '逻辑回归', type: ENTITY_TYPES.MODEL, desc: '用于二分类的线性模型' },
  { id: 'decision_tree', label: '决策树', type: ENTITY_TYPES.MODEL, desc: '基于特征分割的树形分类模型' },
  { id: 'kmeans', label: 'K-均值聚类', type: ENTITY_TYPES.MODEL, desc: '基于距离的聚类算法' },
  { id: 'pca', label: 'PCA', type: ENTITY_TYPES.MODEL, desc: '主成分分析降维方法' },
  
  // 深度学习模型
  { id: 'neural_network', label: '神经网络', type: ENTITY_TYPES.MODEL, desc: '模拟生物神经元的计算模型' },
  { id: 'cnn', label: 'CNN', type: ENTITY_TYPES.MODEL, desc: '卷积神经网络，擅长处理图像数据' },
  { id: 'rnn', label: 'RNN', type: ENTITY_TYPES.MODEL, desc: '循环神经网络，适合处理序列数据' },
  { id: 'transformer', label: 'Transformer', type: ENTITY_TYPES.MODEL, desc: '基于注意力机制的神经网络架构' },
  { id: 'bert', label: 'BERT', type: ENTITY_TYPES.MODEL, desc: '双向编码器表示的预训练语言模型' },
  { id: 'gpt', label: 'GPT', type: ENTITY_TYPES.MODEL, desc: '生成式预训练Transformer模型' },
  
  // 推荐系统模型
  { id: 'collaborative_filtering', label: '协同过滤', type: ENTITY_TYPES.MODEL, desc: '基于用户或物品相似性的推荐方法' },
  { id: 'matrix_factorization', label: '矩阵分解', type: ENTITY_TYPES.MODEL, desc: '将用户-物品矩阵分解为低维因子' },
  { id: 'wide_deep', label: 'Wide & Deep', type: ENTITY_TYPES.MODEL, desc: '结合记忆和泛化能力的推荐模型' },
  { id: 'deepfm', label: 'DeepFM', type: ENTITY_TYPES.MODEL, desc: '结合FM和深度学习的推荐模型' },
  
  // ===== 专业领域 =====
  { id: 'nlp', label: '自然语言处理', type: ENTITY_TYPES.DOMAIN, desc: '让计算机理解和生成人类语言的技术' },
  { id: 'cv', label: '计算机视觉', type: ENTITY_TYPES.DOMAIN, desc: '让计算机理解和分析视觉信息的技术' },
  { id: 'recsys', label: '推荐系统', type: ENTITY_TYPES.DOMAIN, desc: '为用户推荐个性化内容的系统' },
  { id: 'operations_research', label: '运筹优化', type: ENTITY_TYPES.DOMAIN, desc: '使用数学方法解决决策优化问题' },
  
  // ===== 任务类型 =====
  // 监督学习任务
  { id: 'classification', label: '分类', type: ENTITY_TYPES.TASK, desc: '预测离散类别标签的任务' },
  { id: 'regression', label: '回归', type: ENTITY_TYPES.TASK, desc: '预测连续数值的任务' },
  
  // 无监督学习任务
  { id: 'clustering', label: '聚类', type: ENTITY_TYPES.TASK, desc: '将数据分组的无监督任务' },
  { id: 'association_rule_learning', label: '关联规则学习', type: ENTITY_TYPES.TASK, desc: '发现数据项目间关系的任务' },
  { id: 'dimensionality_reduction', label: '降维', type: ENTITY_TYPES.TASK, desc: '减少特征数量的任务' },
  
  // NLP任务
  { id: 'text_classification', label: '文本分类', type: ENTITY_TYPES.TASK, desc: '为文本分配类别标签的任务' },
  { id: 'named_entity_recognition', label: '命名实体识别', type: ENTITY_TYPES.TASK, desc: '识别文本中特定实体的任务' },
  { id: 'machine_translation', label: '机器翻译', type: ENTITY_TYPES.TASK, desc: '将文本从一种语言翻译成另一种语言' },
  { id: 'natural_language_understanding', label: '自然语言理解', type: ENTITY_TYPES.TASK, desc: '从文本中提取深层语义的任务' },
  { id: 'natural_language_generation', label: '自然语言生成', type: ENTITY_TYPES.TASK, desc: '生成自然流畅文本的任务' },
  { id: 'sentiment_analysis', label: '情感分析', type: ENTITY_TYPES.TASK, desc: '判断文本情感倾向的任务' },
  
  // CV任务
  { id: 'image_classification', label: '图像分类', type: ENTITY_TYPES.TASK, desc: '为整张图片分配类别标签的任务' },
  { id: 'object_detection', label: '目标检测', type: ENTITY_TYPES.TASK, desc: '在图像中定位和识别物体的任务' },
  { id: 'image_segmentation', label: '图像分割', type: ENTITY_TYPES.TASK, desc: '对图像中每个像素进行分类的任务' },
  
  // 推荐系统任务
  { id: 'recall', label: '召回', type: ENTITY_TYPES.TASK, desc: '从大量候选中快速筛选的推荐阶段' },
  { id: 'ranking', label: '排序', type: ENTITY_TYPES.TASK, desc: '对候选物品精确打分排序的阶段' },
  { id: 'reranking', label: '重排', type: ENTITY_TYPES.TASK, desc: '考虑多样性等因素调整排序的阶段' },
  
  // ===== 技术工具 =====
  // 编程语言
  { id: 'python', label: 'Python', type: ENTITY_TYPES.TECHNOLOGY, desc: '机器学习领域的主流编程语言' },
  { id: 'cpp', label: 'C++', type: ENTITY_TYPES.TECHNOLOGY, desc: '高性能计算的系统级编程语言' },
  { id: 'java', label: 'Java', type: ENTITY_TYPES.TECHNOLOGY, desc: '企业级应用的面向对象编程语言' },
  
  // 机器学习框架
  { id: 'tensorflow', label: 'TensorFlow', type: ENTITY_TYPES.TECHNOLOGY, desc: 'Google开发的深度学习框架' },
  { id: 'pytorch', label: 'PyTorch', type: ENTITY_TYPES.TECHNOLOGY, desc: 'Facebook开发的深度学习框架' },
  { id: 'scikit_learn', label: 'Scikit-learn', type: ENTITY_TYPES.TECHNOLOGY, desc: '传统机器学习的Python库' },
  
  // 大数据技术
  { id: 'apache_spark', label: 'Apache Spark', type: ENTITY_TYPES.TECHNOLOGY, desc: '分布式大数据处理框架' },
  { id: 'mysql', label: 'MySQL', type: ENTITY_TYPES.TECHNOLOGY, desc: '开源关系型数据库管理系统' },
  { id: 'postgresql', label: 'PostgreSQL', type: ENTITY_TYPES.TECHNOLOGY, desc: '高级开源关系型数据库' },
  { id: 'redis', label: 'Redis', type: ENTITY_TYPES.TECHNOLOGY, desc: '内存数据结构存储系统' },
  { id: 'mongodb', label: 'MongoDB', type: ENTITY_TYPES.TECHNOLOGY, desc: '面向文档的NoSQL数据库' },
  
  // 容器化和编排
  { id: 'docker', label: 'Docker', type: ENTITY_TYPES.TECHNOLOGY, desc: '容器化部署技术' },
  { id: 'kubernetes', label: 'Kubernetes', type: ENTITY_TYPES.TECHNOLOGY, desc: '容器编排平台' },
  
  // MLOps工具
  { id: 'git', label: 'Git', type: ENTITY_TYPES.TECHNOLOGY, desc: '分布式版本控制系统' },
  { id: 'dvc', label: 'DVC', type: ENTITY_TYPES.TECHNOLOGY, desc: '数据版本控制工具' },
  { id: 'mlflow', label: 'MLflow', type: ENTITY_TYPES.TECHNOLOGY, desc: '机器学习生命周期管理平台' },
  
  // 优化求解器
  { id: 'cplex', label: 'CPLEX', type: ENTITY_TYPES.TECHNOLOGY, desc: 'IBM的商业优化求解器' },
  { id: 'gurobi', label: 'Gurobi', type: ENTITY_TYPES.TECHNOLOGY, desc: '高性能的数学优化求解器' },
  
  // ===== 评估指标 =====
  // 分类指标
  { id: 'accuracy', label: '准确率', type: ENTITY_TYPES.METRIC, desc: '分类正确的样本占总样本的比例' },
  { id: 'precision', label: '精确率', type: ENTITY_TYPES.METRIC, desc: '预测为正例中真正例的比例' },
  { id: 'recall', label: '召回率', type: ENTITY_TYPES.METRIC, desc: '真正例中被预测为正例的比例' },
  { id: 'f1_score', label: 'F1分数', type: ENTITY_TYPES.METRIC, desc: '精确率和召回率的调和平均' },
  { id: 'roc_auc', label: 'ROC AUC', type: ENTITY_TYPES.METRIC, desc: 'ROC曲线下面积，衡量分类性能' },
  
  // 回归指标
  { id: 'mae', label: 'MAE', type: ENTITY_TYPES.METRIC, desc: '平均绝对误差，衡量回归模型性能' },
  { id: 'rmse', label: 'RMSE', type: ENTITY_TYPES.METRIC, desc: '均方根误差，衡量回归模型性能' },
  
  // 推荐系统指标
  { id: 'auc', label: 'AUC', type: ENTITY_TYPES.METRIC, desc: '曲线下面积，用于排序任务评估' },
  { id: 'gauc', label: 'GAUC', type: ENTITY_TYPES.METRIC, desc: '分组AUC，衡量个性化推荐效果' },
  
  // ===== 职业级别 =====
  { id: 'junior_engineer', label: '初级算法工程师', type: ENTITY_TYPES.CAREER_LEVEL, desc: '0-2年经验，专注执行和学习' },
  { id: 'senior_engineer', label: '高级算法工程师', type: ENTITY_TYPES.CAREER_LEVEL, desc: '2-5年经验，能独立负责复杂项目' },
  { id: 'expert_engineer', label: '专家算法工程师', type: ENTITY_TYPES.CAREER_LEVEL, desc: '5-10年经验，技术权威和创新者' },
  { id: 'tech_lead', label: '技术主管', type: ENTITY_TYPES.CAREER_LEVEL, desc: '技术和管理并重的领导角色' },
  
  // ===== 软技能 =====
  { id: 'communication', label: '沟通能力', type: ENTITY_TYPES.SKILL, desc: '清晰表达技术方案和与团队协作的能力' },
  { id: 'teamwork', label: '团队协作', type: ENTITY_TYPES.SKILL, desc: '在多学科团队中高效工作的能力' },
  { id: 'leadership', label: '领导力', type: ENTITY_TYPES.SKILL, desc: '影响和指导他人的能力' },
  { id: 'strategic_thinking', label: '战略思维', type: ENTITY_TYPES.SKILL, desc: '从高层次理解业务目标的能力' },
  { id: 'problem_solving', label: '问题解决', type: ENTITY_TYPES.SKILL, desc: '分析和解决复杂问题的能力' },
  { id: 'mentoring', label: '辅导', type: ENTITY_TYPES.SKILL, desc: '指导和培养他人的能力' },
  { id: 'lifelong_learning', label: '终身学习', type: ENTITY_TYPES.SKILL, desc: '持续学习和适应新技术的能力' },
  { id: 'adaptability', label: '适应性', type: ENTITY_TYPES.SKILL, desc: '快速适应变化环境的能力' },
  
  // ===== 数据处理概念 =====
  { id: 'data_collection', label: '数据收集', type: ENTITY_TYPES.CONCEPT, desc: '获取用于机器学习的原始数据' },
  { id: 'data_preparation', label: '数据准备', type: ENTITY_TYPES.CONCEPT, desc: '清洗和整理数据用于模型训练' },
  { id: 'feature_selection', label: '特征选择', type: ENTITY_TYPES.CONCEPT, desc: '选择最有用特征的过程' },
  { id: 'normalization', label: '归一化', type: ENTITY_TYPES.CONCEPT, desc: '将数据缩放到特定范围的处理' },
  { id: 'standardization', label: '标准化', type: ENTITY_TYPES.CONCEPT, desc: '将数据转换为标准正态分布' },
  { id: 'one_hot_encoding', label: '独热编码', type: ENTITY_TYPES.CONCEPT, desc: '将类别变量转换为二进制向量' },
  { id: 'training_set', label: '训练集', type: ENTITY_TYPES.CONCEPT, desc: '用于训练模型的数据集' },
  { id: 'validation_set', label: '验证集', type: ENTITY_TYPES.CONCEPT, desc: '用于调整超参数的数据集' },
  { id: 'test_set', label: '测试集', type: ENTITY_TYPES.CONCEPT, desc: '用于最终评估模型的数据集' },
  { id: 'k_fold_cv', label: 'K折交叉验证', type: ENTITY_TYPES.CONCEPT, desc: '将数据分为K份进行验证的方法' },
  
  // ===== 文本处理技术 =====
  { id: 'tokenization', label: '分词', type: ENTITY_TYPES.CONCEPT, desc: '将文本切分成词元的过程' },
  { id: 'word_embedding', label: '词嵌入', type: ENTITY_TYPES.CONCEPT, desc: '将词语映射到向量空间的技术' },
  { id: 'word2vec', label: 'Word2Vec', type: ENTITY_TYPES.MODEL, desc: '学习词向量表示的神经网络模型' },
  { id: 'tfidf', label: 'TF-IDF', type: ENTITY_TYPES.CONCEPT, desc: '词频-逆文档频率的文本表示方法' },
  { id: 'bag_of_words', label: '词袋模型', type: ENTITY_TYPES.CONCEPT, desc: '将文本表示为词汇集合的方法' },
  { id: 'contextual_embedding', label: '上下文嵌入', type: ENTITY_TYPES.CONCEPT, desc: '根据上下文动态生成的词向量' },
  
  // ===== 图像处理技术 =====
  { id: 'convolution', label: '卷积', type: ENTITY_TYPES.CONCEPT, desc: '图像处理中的特征提取操作' },
  { id: 'pooling', label: '池化', type: ENTITY_TYPES.CONCEPT, desc: '降采样操作，减少特征图尺寸' },
  { id: 'filter', label: '滤波器', type: ENTITY_TYPES.CONCEPT, desc: '卷积操作中的参数矩阵' },
  { id: 'kernel', label: '卷积核', type: ENTITY_TYPES.CONCEPT, desc: '卷积操作的核心计算单元' },
  { id: 'edge_detection', label: '边缘检测', type: ENTITY_TYPES.CONCEPT, desc: '识别图像中边缘特征的技术' },
  { id: 'texture_extraction', label: '纹理提取', type: ENTITY_TYPES.CONCEPT, desc: '提取图像纹理特征的技术' },
  
  // ===== 推荐系统技术 =====
  { id: 'content_based_filtering', label: '基于内容的推荐', type: ENTITY_TYPES.ALGORITHM, desc: '根据物品内容属性进行推荐' },
  { id: 'user_based_cf', label: '基于用户的协同过滤', type: ENTITY_TYPES.ALGORITHM, desc: '基于用户相似性的推荐方法' },
  { id: 'item_based_cf', label: '基于物品的协同过滤', type: ENTITY_TYPES.ALGORITHM, desc: '基于物品相似性的推荐方法' },
  
  // ===== 注意力机制 =====
  { id: 'attention_mechanism', label: '注意力机制', type: ENTITY_TYPES.CONCEPT, desc: '让模型关注输入中重要部分的技术' },
  { id: 'self_attention', label: '自注意力机制', type: ENTITY_TYPES.CONCEPT, desc: '计算序列内部元素间关系的注意力' },
  { id: 'multi_head_attention', label: '多头注意力', type: ENTITY_TYPES.CONCEPT, desc: '并行计算多个注意力的机制' },
  
  // ===== 序列模型 =====
  { id: 'seq2seq', label: '序列到序列模型', type: ENTITY_TYPES.MODEL, desc: '将输入序列映射到输出序列的架构' },
  { id: 'encoder_decoder', label: '编码器-解码器', type: ENTITY_TYPES.CONCEPT, desc: 'Seq2Seq模型的基本架构框架' },
  
  // ===== 应用领域 =====
  { id: 'fraud_detection', label: '欺诈检测', type: ENTITY_TYPES.TASK, desc: '识别欺诈行为的分类任务' },
  { id: 'customer_churn', label: '客户流失预测', type: ENTITY_TYPES.TASK, desc: '预测客户是否会流失的任务' },
  { id: 'stock_prediction', label: '股价预测', type: ENTITY_TYPES.TASK, desc: '预测股票价格走势的回归任务' },
  { id: 'speech_recognition', label: '语音识别', type: ENTITY_TYPES.TASK, desc: '将语音转换为文本的任务' },
  { id: 'autonomous_driving', label: '自动驾驶', type: ENTITY_TYPES.TASK, desc: '无人驾驶汽车的决策控制任务' },
  { id: 'game_ai', label: '游戏AI', type: ENTITY_TYPES.TASK, desc: '在游戏中实现智能决策的任务' },
  { id: 'robot_control', label: '机器人控制', type: ENTITY_TYPES.TASK, desc: '控制机器人行为的任务' },
  { id: 'resource_scheduling', label: '资源调度', type: ENTITY_TYPES.TASK, desc: '优化资源分配的调度任务' },
  { id: 'shortest_path', label: '最短路径', type: ENTITY_TYPES.TASK, desc: '在图中寻找两点间最短路径的问题' },
  
  // ===== 文档中缺失的重要实体 =====
  // 数据库系统
  { id: 'database_index', label: '数据库索引', type: ENTITY_TYPES.CONCEPT, desc: '提高数据库查询性能的数据结构' },
  { id: 'cache_system', label: '缓存系统', type: ENTITY_TYPES.CONCEPT, desc: '提高数据访问速度的系统' },
  
  // 数据分布和漂移
  { id: 'data_distribution', label: '数据分布', type: ENTITY_TYPES.CONCEPT, desc: '数据在特征空间中的分布情况' },
  
  // 硬件相关
  { id: 'gpu', label: 'GPU', type: ENTITY_TYPES.TECHNOLOGY, desc: '图形处理器，用于并行计算' },
  { id: 'parallel_computing', label: '并行计算', type: ENTITY_TYPES.CONCEPT, desc: '同时执行多个计算任务的技术' },
  { id: 'hardware_interaction', label: '硬件交互', type: ENTITY_TYPES.CONCEPT, desc: '算法与底层硬件的交互关系' },
  
  // 数据科学相关
  { id: 'jupyter_notebook', label: 'Jupyter Notebook', type: ENTITY_TYPES.TECHNOLOGY, desc: '交互式数据科学开发环境' },
  { id: 'data_visualization', label: '数据可视化', type: ENTITY_TYPES.CONCEPT, desc: '将数据转化为图形化表示的技术' },
  
  // 软件工程实践
  { id: 'code_review', label: '代码审查', type: ENTITY_TYPES.CONCEPT, desc: '团队成员互相检查代码质量的实践' },
  { id: 'design_review', label: '设计审查', type: ENTITY_TYPES.CONCEPT, desc: '对系统设计方案的评审过程' },
  { id: 'technical_sharing', label: '技术分享', type: ENTITY_TYPES.CONCEPT, desc: '在团队内分享技术知识的实践' },
  { id: 'best_practices', label: '最佳实践', type: ENTITY_TYPES.CONCEPT, desc: '经过验证的优秀工程实践' },
  
  // 业务相关
  { id: 'product_manager', label: '产品经理', type: ENTITY_TYPES.CONCEPT, desc: '负责产品规划和管理的角色' },
  { id: 'business_analyst', label: '业务分析师', type: ENTITY_TYPES.CONCEPT, desc: '分析业务需求和数据的专业人员' },
  { id: 'stakeholder', label: '利益相关者', type: ENTITY_TYPES.CONCEPT, desc: '对项目成果有利益关系的人员' },
  
  // 性能相关
  { id: 'latency', label: '延迟', type: ENTITY_TYPES.METRIC, desc: '系统响应请求的时间' },
  { id: 'throughput', label: '吞吐量', type: ENTITY_TYPES.METRIC, desc: '系统单位时间处理的请求数量' },
  { id: 'resource_consumption', label: '资源消耗', type: ENTITY_TYPES.METRIC, desc: '系统运行所需的计算资源' },
  
  // 算法理论
  { id: 'np_hard', label: 'NP-hard', type: ENTITY_TYPES.CONCEPT, desc: '计算复杂度极高的问题类型' },
  { id: 'approximation_algorithm', label: '近似算法', type: ENTITY_TYPES.CONCEPT, desc: '在合理时间内找到近似解的算法' },
  
  // 机器学习理论
  { id: 'bias_variance_tradeoff', label: '偏差-方差权衡', type: ENTITY_TYPES.CONCEPT, desc: '机器学习中的核心理论问题' },
  { id: 'no_free_lunch', label: '没有免费的午餐', type: ENTITY_TYPES.CONCEPT, desc: '没有一种算法在所有问题上都最优的定理' },
  
  // 数据结构进阶概念
  { id: 'memory_layout', label: '内存布局', type: ENTITY_TYPES.CONCEPT, desc: '数据在内存中的组织方式' },
  { id: 'pointer_connection', label: '指针连接', type: ENTITY_TYPES.CONCEPT, desc: '通过指针连接数据元素的方式' },
  { id: 'insertion_cost', label: '插入成本', type: ENTITY_TYPES.CONCEPT, desc: '在数据结构中插入元素的计算代价' },
  { id: 'deletion_cost', label: '删除成本', type: ENTITY_TYPES.CONCEPT, desc: '在数据结构中删除元素的计算代价' },
  
  // 机器学习工程
  { id: 'feature_importance', label: '特征重要性', type: ENTITY_TYPES.CONCEPT, desc: '衡量特征对模型预测的贡献程度' },
  { id: 'model_interpretability_methods', label: '模型解释方法', type: ENTITY_TYPES.CONCEPT, desc: '理解模型决策过程的各种技术' },
  { id: 'feature_attribution', label: '特征归因', type: ENTITY_TYPES.CONCEPT, desc: '分析各个特征对预测结果的影响' },

  // MLOps - 补充实体
  { id: 'devops', label: 'DevOps', type: ENTITY_TYPES.CONCEPT, desc: '软件开发与信息技术运维的实践集合' },
  { id: 'version_control', label: '版本控制', type: ENTITY_TYPES.CONCEPT, desc: '管理代码和数据变更的系统' },
  { id: 'experiment_tracking', label: '实验跟踪', type: ENTITY_TYPES.CONCEPT, desc: '记录和管理机器学习实验的实践' },
  { id: 'model_registry', label: '模型注册', type: ENTITY_TYPES.CONCEPT, desc: '存储和版本化训练好的模型的系统' },
  { id: 'ci_cd', label: 'CI/CD', type: ENTITY_TYPES.CONCEPT, desc: '持续集成/持续部署' },
  { id: 'automated_pipeline', label: '自动化流水线', type: ENTITY_TYPES.CONCEPT, desc: '自动化机器学习工作流' },
  { id: 'rest_api', label: 'REST API', type: ENTITY_TYPES.TECHNOLOGY, desc: '一种用于Web服务的架构风格' },
  { id: 'batch_prediction', label: '批量预测', type: ENTITY_TYPES.CONCEPT, desc: '对大量数据进行离线预测' },
  { id: 'online_serving', label: '在线服务', type: ENTITY_TYPES.CONCEPT, desc: '提供实时模型预测服务' },
  { id: 'model_fairness', label: '模型公平性', type: ENTITY_TYPES.CONCEPT, desc: '确保模型对不同群体没有偏见' },
  { id: 'model_interpretability', label: '模型可解释性', type: ENTITY_TYPES.CONCEPT, desc: '理解模型为何做出特定预测' },
  { id: 'compliance', label: '合规性', type: ENTITY_TYPES.CONCEPT, desc: '遵守相关法规和政策' },
  { id: 'bias_detection', label: '偏见检测', type: ENTITY_TYPES.CONCEPT, desc: '检测数据和模型中的偏见' },
];

// 生成算法知识图谱关系边
function generateEdges(): EdgeData[] {
  return [
    // 基础学科关系
    { id: 'e0', source: 'computer_science', target: 'dsa', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e0_1', source: 'computer_science', target: 'ml', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e0_2', source: 'mathematics', target: 'ml', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '支持' },
    { id: 'e0_3', source: 'dsa', target: 'ml', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '支持' },
    
    // 复杂度分析关系
    { id: 'e0_4', source: 'complexity_analysis', target: 'time_complexity', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e0_5', source: 'complexity_analysis', target: 'space_complexity', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e0_6', source: 'big_o_notation', target: 'complexity_analysis', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '用于' },
    { id: 'e0_7', source: 'complexity_analysis', target: 'best_case', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e0_8', source: 'complexity_analysis', target: 'average_case', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e0_9', source: 'complexity_analysis', target: 'worst_case', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    
    // 机器学习包含关系
    { id: 'e1', source: 'ml', target: 'supervised_learning', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e2', source: 'ml', target: 'unsupervised_learning', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e3', source: 'ml', target: 'reinforcement_learning', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e4', source: 'ml', target: 'deep_learning', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    
    // 学习范式与任务的关系
    { id: 'e5', source: 'supervised_learning', target: 'classification', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e6', source: 'supervised_learning', target: 'regression', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e7', source: 'unsupervised_learning', target: 'clustering', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    
    // 模型应用关系
    { id: 'e8', source: 'cnn', target: 'cv', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e9', source: 'rnn', target: 'nlp', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e10', source: 'transformer', target: 'nlp', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e11', source: 'cnn', target: 'object_detection', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '用于' },
    { id: 'e12', source: 'transformer', target: 'machine_translation', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '用于' },
    
    // 模型演进关系
    { id: 'e13', source: 'rnn', target: 'transformer', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '发展为' },
    
    // 算法与数据结构关系
    { id: 'e14', source: 'quick_sort', target: 'array', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e15', source: 'binary_search', target: 'array', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e16', source: 'dijkstra', target: 'graph', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    
    // 技术实现关系
    { id: 'e17', source: 'python', target: 'ml', edgeType: RELATIONSHIP_TYPES.IMPLEMENTS, label: '实现' },
    { id: 'e18', source: 'tensorflow', target: 'deep_learning', edgeType: RELATIONSHIP_TYPES.IMPLEMENTS, label: '实现' },
    { id: 'e19', source: 'pytorch', target: 'deep_learning', edgeType: RELATIONSHIP_TYPES.IMPLEMENTS, label: '实现' },
    { id: 'e20', source: 'docker', target: 'ml', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '支持' },
    
    // 评估指标关系
    { id: 'e21', source: 'accuracy', target: 'classification', edgeType: RELATIONSHIP_TYPES.EVALUATES, label: '度量' },
    { id: 'e22', source: 'precision', target: 'classification', edgeType: RELATIONSHIP_TYPES.EVALUATES, label: '度量' },
    { id: 'e23', source: 'recall', target: 'classification', edgeType: RELATIONSHIP_TYPES.EVALUATES, label: '度量' },
    { id: 'e24', source: 'rmse', target: 'regression', edgeType: RELATIONSHIP_TYPES.EVALUATES, label: '度量' },
    
    // 职业发展关系
    { id: 'e25', source: 'junior_engineer', target: 'senior_engineer', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '发展为' },
    { id: 'e26', source: 'senior_engineer', target: 'expert_engineer', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '发展为' },
    { id: 'e27', source: 'expert_engineer', target: 'tech_lead', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '发展为' },
    
    // 技能需求关系
    { id: 'e28', source: 'senior_engineer', target: 'communication', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e29', source: 'senior_engineer', target: 'teamwork', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e30', source: 'expert_engineer', target: 'leadership', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e31', source: 'tech_lead', target: 'strategic_thinking', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    
    // 模型对比关系
    { id: 'e32', source: 'cnn', target: 'rnn', edgeType: RELATIONSHIP_TYPES.COMPARES_WITH, label: '对比' },
    { id: 'e33', source: 'supervised_learning', target: 'unsupervised_learning', edgeType: RELATIONSHIP_TYPES.COMPARES_WITH, label: '对比' },
    
    // 算法优化关系
    { id: 'e34', source: 'quick_sort', target: 'merge_sort', edgeType: RELATIONSHIP_TYPES.COMPARES_WITH, label: '对比' },
    { id: 'e35', source: 'binary_search', target: 'hash_table', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '优化' },
    
    // 深度学习模型关系
    { id: 'e36', source: 'deep_learning', target: 'cnn', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e37', source: 'deep_learning', target: 'rnn', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e38', source: 'deep_learning', target: 'transformer', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    
    // 传统机器学习模型
    { id: 'e39', source: 'supervised_learning', target: 'linear_regression', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e40', source: 'supervised_learning', target: 'decision_tree', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    
    // MLOps关系
    { id: 'e20_15', source: 'mlops', target: 'devops', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e20_16', source: 'mlops', target: 'version_control', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e20_17', source: 'mlops', target: 'experiment_tracking', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e20_18', source: 'mlops', target: 'model_registry', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e20_19', source: 'mlops', target: 'ci_cd', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e20_20', source: 'mlops', target: 'automated_pipeline', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e20_21', source: 'git', target: 'version_control', edgeType: RELATIONSHIP_TYPES.IMPLEMENTS, label: '实现' },
    { id: 'e20_22', source: 'dvc', target: 'version_control', edgeType: RELATIONSHIP_TYPES.IMPLEMENTS, label: '实现' },
    { id: 'e20_23', source: 'mlflow', target: 'experiment_tracking', edgeType: RELATIONSHIP_TYPES.IMPLEMENTS, label: '实现' },
    { id: 'e20_24', source: 'mlflow', target: 'model_registry', edgeType: RELATIONSHIP_TYPES.IMPLEMENTS, label: '实现' },
    { id: 'e20_25', source: 'model_deployment', target: 'rest_api', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e20_26', source: 'model_deployment', target: 'batch_prediction', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '支持' },
    { id: 'e20_27', source: 'model_deployment', target: 'online_serving', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '支持' },
    { id: 'e20_28', source: 'model_monitoring', target: 'data_drift', edgeType: RELATIONSHIP_TYPES.EVALUATES, label: '检测' },
    { id: 'e20_29', source: 'model_monitoring', target: 'concept_drift', edgeType: RELATIONSHIP_TYPES.EVALUATES, label: '检测' },
    { id: 'e20_30', source: 'model_monitoring', target: 'model_fairness', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '保证' },
    { id: 'e20_31', source: 'model_monitoring', target: 'model_interpretability', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '保证' },
    { id: 'e20_32', source: 'model_monitoring', target: 'compliance', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '保证' },
    { id: 'e20_33', source: 'model_monitoring', target: 'bias_detection', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    
    // 领域特定任务
    { id: 'e41', source: 'cv', target: 'object_detection', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e42', source: 'nlp', target: 'machine_translation', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },

    // ===== 新增关系 =====

    // 数据结构关系
    { id: 'e43', source: 'linear_structure', target: 'array', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e44', source: 'linear_structure', target: 'linked_list', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e45', source: 'linear_structure', target: 'stack', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e46', source: 'linear_structure', target: 'queue', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e47', source: 'nonlinear_structure', target: 'binary_tree', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e48', source: 'nonlinear_structure', target: 'heap', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e49', source: 'nonlinear_structure', target: 'graph', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e50', source: 'stack', target: 'array', edgeType: RELATIONSHIP_TYPES.IMPLEMENTS, label: '可由...实现' },
    { id: 'e51', source: 'queue', target: 'linked_list', edgeType: RELATIONSHIP_TYPES.IMPLEMENTS, label: '可由...实现' },
    { id: 'e52', source: 'avl_tree', target: 'binary_search_tree', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e53', source: 'red_black_tree', target: 'binary_search_tree', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e54', source: 'binary_search_tree', target: 'tree_degeneration', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '导致' },
    { id: 'e55', source: 'avl_tree', target: 'tree_degeneration', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '防止' },
    { id: 'e56', source: 'priority_queue', target: 'heap', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e57', source: 'hash_table', target: 'hash_function', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e58', source: 'hash_table', target: 'hash_collision', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '导致' },

    // 算法与范式/数据结构关系
    { id: 'e59', source: 'merge_sort', target: 'divide_conquer', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e60', source: 'quick_sort', target: 'divide_conquer', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e61', source: 'dijkstra', target: 'greedy_algorithm', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e62', source: 'prim', target: 'greedy_algorithm', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e63', source: 'dynamic_programming', target: 'optimal_substructure', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e64', source: 'dynamic_programming', target: 'overlapping_subproblems', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e65', source: 'heap_sort', target: 'heap', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e66', source: 'bfs', target: 'queue', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e67', source: 'dfs', target: 'stack', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e68', source: 'binary_search', target: 'divide_conquer', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    
    // 机器学习/深度学习模型关系
    { id: 'e69', source: 'logistic_regression', target: 'classification', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '解决' },
    { id: 'e70', source: 'kmeans', target: 'clustering', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '解决' },
    { id: 'e71', source: 'pca', target: 'dimensionality_reduction', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '解决' },
    { id: 'e72', source: 'bert', target: 'transformer', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e73', source: 'gpt', target: 'transformer', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e74', source: 'transformer', target: 'self_attention', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e75', source: 'bert', target: 'contextual_embedding', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '实现' },
    { id: 'e76', source: 'model_lifecycle', target: 'feature_engineering', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e77', source: 'model_lifecycle', target: 'model_training', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e78', source: 'model_lifecycle', target: 'model_evaluation', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e79', source: 'overfitting', target: 'generalization', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '阻碍' },
    { id: 'e80', source: 'deep_learning', target: 'gradient_vanishing', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '导致' },

    // 专业领域与技术/任务关系
    { id: 'e81', source: 'nlp', target: 'word_embedding', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e82', source: 'cv', target: 'convolution', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e83', source: 'recsys', target: 'collaborative_filtering', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e84', source: 'recsys', target: 'matrix_factorization', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e85', source: 'operations_research', target: 'linear_programming', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },

    // 技术与概念关系
    { id: 'e86', source: 'deep_learning', target: 'gpu', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e87', source: 'gpu', target: 'parallel_computing', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '实现' },
    { id: 'e88', source: 'scikit_learn', target: 'ml', edgeType: RELATIONSHIP_TYPES.IMPLEMENTS, label: '实现' },

    // ===== 强化学习关系 =====
    { id: 'e89', source: 'reinforcement_learning', target: 'agent', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e90', source: 'reinforcement_learning', target: 'environment', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e91', source: 'reinforcement_learning', target: 'action', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e92', source: 'reinforcement_learning', target: 'reward', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e93', source: 'reinforcement_learning', target: 'penalty', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e94', source: 'reinforcement_learning', target: 'policy', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e95', source: 'reinforcement_learning', target: 'state', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e96', source: 'agent', target: 'environment', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '与...交互' },
    { id: 'e97', source: 'agent', target: 'action', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '执行' },
    { id: 'e98', source: 'agent', target: 'policy', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '遵循' },
    { id: 'e99', source: 'environment', target: 'reward', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '给予' },
    { id: 'e100', source: 'environment', target: 'penalty', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '给予' },
    { id: 'e101', source: 'environment', target: 'state', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '维护' },
    { id: 'e102', source: 'policy', target: 'state', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e103', source: 'policy', target: 'action', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '决定' },
    { id: 'e104', source: 'q_learning', target: 'reinforcement_learning', edgeType: RELATIONSHIP_TYPES.IS_A, label: '属于' },
    { id: 'e105', source: 'policy_gradient', target: 'reinforcement_learning', edgeType: RELATIONSHIP_TYPES.IS_A, label: '属于' },
    { id: 'e106', source: 'actor_critic', target: 'reinforcement_learning', edgeType: RELATIONSHIP_TYPES.IS_A, label: '属于' },
    { id: 'e107', source: 'reinforcement_learning', target: 'game_ai', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e108', source: 'reinforcement_learning', target: 'robot_control', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e109', source: 'reinforcement_learning', target: 'autonomous_driving', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },

    // ===== 数据结构特性关系 =====
    { id: 'e110', source: 'array', target: 'cache_friendly', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '具有' },
    { id: 'e111', source: 'array', target: 'random_access', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '支持' },
    { id: 'e112', source: 'linked_list', target: 'dynamic_insertion', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '支持' },
    { id: 'e113', source: 'linked_list', target: 'dynamic_deletion', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '支持' },
    { id: 'e114', source: 'array', target: 'memory_layout', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '连续内存' },
    { id: 'e115', source: 'linked_list', target: 'pointer_connection', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e116', source: 'cache_friendly', target: 'hardware_interaction', edgeType: RELATIONSHIP_TYPES.IS_A, label: '属于' },
    { id: 'e117', source: 'array', target: 'linked_list', edgeType: RELATIONSHIP_TYPES.COMPARES_WITH, label: '对比' },

    // ===== 树相关关系 =====
    { id: 'e118', source: 'binary_tree', target: 'binary_search_tree', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e119', source: 'avl_tree', target: 'tree_balance', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '维持' },
    { id: 'e120', source: 'red_black_tree', target: 'tree_balance', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '维持' },
    { id: 'e121', source: 'avl_tree', target: 'tree_rotation', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e122', source: 'red_black_tree', target: 'tree_rotation', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e123', source: 'tree_rotation', target: 'tree_balance', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '维持' },
    { id: 'e124', source: 'heap', target: 'heap_property', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e125', source: 'heap', target: 'complete_binary_tree', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e126', source: 'heap', target: 'array', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '用...实现' },
    { id: 'e127', source: 'avl_tree', target: 'red_black_tree', edgeType: RELATIONSHIP_TYPES.COMPARES_WITH, label: '对比' },
    { id: 'e128', source: 'binary_search_tree', target: 'hash_table', edgeType: RELATIONSHIP_TYPES.COMPARES_WITH, label: '对比' },

    // ===== 图结构关系 =====
    { id: 'e129', source: 'graph', target: 'vertices', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e130', source: 'graph', target: 'edges', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e131', source: 'bfs', target: 'graph_traversal', edgeType: RELATIONSHIP_TYPES.IMPLEMENTS, label: '实现' },
    { id: 'e132', source: 'dfs', target: 'graph_traversal', edgeType: RELATIONSHIP_TYPES.IMPLEMENTS, label: '实现' },
    { id: 'e133', source: 'graph', target: 'network_analysis', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e134', source: 'network_analysis', target: 'social_network', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e135', source: 'graph', target: 'knowledge_graph', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e136', source: 'dijkstra', target: 'logistics_routing', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e137', source: 'kruskal', target: 'greedy_algorithm', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e138', source: 'prim', target: 'kruskal', edgeType: RELATIONSHIP_TYPES.COMPARES_WITH, label: '对比' },
    { id: 'e139', source: 'bfs', target: 'dfs', edgeType: RELATIONSHIP_TYPES.COMPARES_WITH, label: '对比' },

    // ===== 排序算法分类关系 =====
    { id: 'e140', source: 'comparison_sort', target: 'bubble_sort', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e141', source: 'comparison_sort', target: 'insertion_sort', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e142', source: 'comparison_sort', target: 'selection_sort', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e143', source: 'comparison_sort', target: 'merge_sort', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e144', source: 'comparison_sort', target: 'quick_sort', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e145', source: 'comparison_sort', target: 'heap_sort', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e146', source: 'non_comparison_sort', target: 'counting_sort', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e147', source: 'non_comparison_sort', target: 'radix_sort', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e148', source: 'non_comparison_sort', target: 'bucket_sort', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e149', source: 'stable_sort', target: 'merge_sort', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e150', source: 'stable_sort', target: 'bubble_sort', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e151', source: 'stable_sort', target: 'insertion_sort', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e152', source: 'in_place_sort', target: 'quick_sort', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e153', source: 'in_place_sort', target: 'heap_sort', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e154', source: 'in_place_sort', target: 'bubble_sort', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e155', source: 'greedy_algorithm', target: 'greedy_choice_property', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e156', source: 'huffman_coding', target: 'greedy_algorithm', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e157', source: 'linear_search', target: 'binary_search', edgeType: RELATIONSHIP_TYPES.COMPARES_WITH, label: '对比' },

    // ===== 机器学习流程关系 =====
    { id: 'e158', source: 'data_preprocessing', target: 'feature_engineering', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '导向' },
    { id: 'e159', source: 'feature_engineering', target: 'model_training', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '导向' },
    { id: 'e160', source: 'model_training', target: 'model_evaluation', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '导向' },
    { id: 'e161', source: 'model_evaluation', target: 'hyperparameter_tuning', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '导向' },
    { id: 'e162', source: 'hyperparameter_tuning', target: 'model_deployment', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '导向' },
    { id: 'e163', source: 'model_deployment', target: 'model_monitoring', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '导向' },
    { id: 'e164', source: 'data_leakage', target: 'model_evaluation', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '影响' },
    { id: 'e165', source: 'data_drift', target: 'data_distribution', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e166', source: 'concept_drift', target: 'data_distribution', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e167', source: 'problem_definition', target: 'data_collection', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '导向' },
    { id: 'e168', source: 'model_lifecycle', target: 'problem_definition', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },

    // ===== 数据处理关系 =====
    { id: 'e169', source: 'data_preprocessing', target: 'data_collection', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e170', source: 'feature_engineering', target: 'feature_selection', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e171', source: 'feature_engineering', target: 'normalization', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e172', source: 'feature_engineering', target: 'standardization', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e173', source: 'feature_engineering', target: 'one_hot_encoding', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e174', source: 'data_preparation', target: 'training_set', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '产生' },
    { id: 'e175', source: 'data_preparation', target: 'validation_set', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '产生' },
    { id: 'e176', source: 'data_preparation', target: 'test_set', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '产生' },
    { id: 'e177', source: 'k_fold_cv', target: 'validation_set', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e178', source: 'k_fold_cv', target: 'model_evaluation', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '用于' },
    { id: 'e179', source: 'normalization', target: 'standardization', edgeType: RELATIONSHIP_TYPES.COMPARES_WITH, label: '对比' },

    // ===== 文本处理关系 =====
    { id: 'e180', source: 'nlp', target: 'tokenization', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e181', source: 'tokenization', target: 'word_embedding', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '导向' },
    { id: 'e182', source: 'word2vec', target: 'word_embedding', edgeType: RELATIONSHIP_TYPES.IMPLEMENTS, label: '实现' },
    { id: 'e183', source: 'bert', target: 'word_embedding', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e184', source: 'tfidf', target: 'bag_of_words', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e185', source: 'contextual_embedding', target: 'word_embedding', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '演进自' },
    { id: 'e186', source: 'nlp', target: 'text_classification', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e187', source: 'nlp', target: 'named_entity_recognition', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e188', source: 'nlp', target: 'sentiment_analysis', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e189', source: 'nlp', target: 'natural_language_understanding', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e190', source: 'nlp', target: 'natural_language_generation', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e191', source: 'text_classification', target: 'classification', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e192', source: 'sentiment_analysis', target: 'text_classification', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },

    // ===== 图像处理关系 =====
    { id: 'e193', source: 'cnn', target: 'convolution', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e194', source: 'cnn', target: 'pooling', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e195', source: 'convolution', target: 'filter', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e196', source: 'convolution', target: 'kernel', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e197', source: 'convolution', target: 'edge_detection', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '实现' },
    { id: 'e198', source: 'convolution', target: 'texture_extraction', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '实现' },
    { id: 'e199', source: 'cv', target: 'image_classification', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e200', source: 'cv', target: 'image_segmentation', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e201', source: 'image_classification', target: 'classification', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },

    // ===== 推荐系统关系 =====
    { id: 'e202', source: 'collaborative_filtering', target: 'user_based_cf', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e203', source: 'collaborative_filtering', target: 'item_based_cf', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e204', source: 'collaborative_filtering', target: 'content_based_filtering', edgeType: RELATIONSHIP_TYPES.COMPARES_WITH, label: '对比' },
    { id: 'e205', source: 'matrix_factorization', target: 'collaborative_filtering', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '增强' },
    { id: 'e206', source: 'wide_deep', target: 'deep_learning', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e207', source: 'deepfm', target: 'deep_learning', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e208', source: 'recsys', target: 'recall', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e209', source: 'recsys', target: 'ranking', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e210', source: 'recsys', target: 'reranking', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e211', source: 'recall', target: 'ranking', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '导向' },
    { id: 'e212', source: 'ranking', target: 'reranking', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '导向' },

    // ===== 注意力机制关系 =====
    { id: 'e213', source: 'attention_mechanism', target: 'self_attention', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e214', source: 'attention_mechanism', target: 'multi_head_attention', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e215', source: 'multi_head_attention', target: 'self_attention', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e216', source: 'transformer', target: 'attention_mechanism', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e217', source: 'transformer', target: 'multi_head_attention', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },

    // ===== 序列模型关系 =====
    { id: 'e218', source: 'seq2seq', target: 'encoder_decoder', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e219', source: 'seq2seq', target: 'rnn', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e220', source: 'seq2seq', target: 'machine_translation', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e221', source: 'seq2seq', target: 'attention_mechanism', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },

    // ===== 优化算法关系 =====
    { id: 'e222', source: 'operations_research', target: 'integer_programming', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e223', source: 'integer_programming', target: 'linear_programming', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e224', source: 'operations_research', target: 'heuristic_algorithm', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e225', source: 'metaheuristic_algorithm', target: 'heuristic_algorithm', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e226', source: 'simulated_annealing', target: 'metaheuristic_algorithm', edgeType: RELATIONSHIP_TYPES.IS_A, label: '属于' },
    { id: 'e227', source: 'genetic_algorithm', target: 'metaheuristic_algorithm', edgeType: RELATIONSHIP_TYPES.IS_A, label: '属于' },
    { id: 'e228', source: 'branch_bound', target: 'integer_programming', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '解决' },
    { id: 'e229', source: 'cplex', target: 'linear_programming', edgeType: RELATIONSHIP_TYPES.IMPLEMENTS, label: '实现' },
    { id: 'e230', source: 'gurobi', target: 'linear_programming', edgeType: RELATIONSHIP_TYPES.IMPLEMENTS, label: '实现' },
    { id: 'e231', source: 'operations_research', target: 'resource_scheduling', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e232', source: 'np_hard', target: 'heuristic_algorithm', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e233', source: 'approximation_algorithm', target: 'np_hard', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '解决' },

    // ===== 超参数调优关系 =====
    { id: 'e234', source: 'grid_search', target: 'hyperparameter_tuning', edgeType: RELATIONSHIP_TYPES.IMPLEMENTS, label: '实现' },
    { id: 'e235', source: 'random_search', target: 'hyperparameter_tuning', edgeType: RELATIONSHIP_TYPES.IMPLEMENTS, label: '实现' },
    { id: 'e236', source: 'bayesian_optimization', target: 'hyperparameter_tuning', edgeType: RELATIONSHIP_TYPES.IMPLEMENTS, label: '实现' },
    { id: 'e237', source: 'grid_search', target: 'random_search', edgeType: RELATIONSHIP_TYPES.COMPARES_WITH, label: '对比' },
    { id: 'e238', source: 'random_search', target: 'bayesian_optimization', edgeType: RELATIONSHIP_TYPES.COMPARES_WITH, label: '对比' },

    // ===== 评估指标细化关系 =====
    { id: 'e239', source: 'f1_score', target: 'precision', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e240', source: 'f1_score', target: 'recall', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e241', source: 'mae', target: 'regression', edgeType: RELATIONSHIP_TYPES.EVALUATES, label: '度量' },
    { id: 'e242', source: 'auc', target: 'classification', edgeType: RELATIONSHIP_TYPES.EVALUATES, label: '度量' },
    { id: 'e243', source: 'gauc', target: 'ranking', edgeType: RELATIONSHIP_TYPES.EVALUATES, label: '度量' },
    { id: 'e244', source: 'latency', target: 'model_deployment', edgeType: RELATIONSHIP_TYPES.EVALUATES, label: '度量' },
    { id: 'e245', source: 'throughput', target: 'model_deployment', edgeType: RELATIONSHIP_TYPES.EVALUATES, label: '度量' },

    // ===== 应用任务关系 =====
    { id: 'e246', source: 'fraud_detection', target: 'classification', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e247', source: 'customer_churn', target: 'classification', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e248', source: 'stock_prediction', target: 'regression', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e249', source: 'speech_recognition', target: 'nlp', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },

    // ===== 职业级别与技能关系（更细化） =====
    { id: 'e250', source: 'junior_engineer', target: 'python', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e251', source: 'junior_engineer', target: 'dsa', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e252', source: 'junior_engineer', target: 'ml', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e253', source: 'junior_engineer', target: 'problem_solving', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e254', source: 'junior_engineer', target: 'lifelong_learning', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e255', source: 'senior_engineer', target: 'deep_learning', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e256', source: 'senior_engineer', target: 'mlops', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e257', source: 'expert_engineer', target: 'mentoring', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e258', source: 'expert_engineer', target: 'adaptability', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e259', source: 'tech_lead', target: 'teamwork', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },

    // ===== 技术栈关系 =====
    { id: 'e260', source: 'jupyter_notebook', target: 'python', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e261', source: 'jupyter_notebook', target: 'data_visualization', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '支持' },
    { id: 'e262', source: 'apache_spark', target: 'python', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '支持' },
    { id: 'e263', source: 'apache_spark', target: 'java', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '支持' },
    { id: 'e264', source: 'kubernetes', target: 'docker', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '管理' },
    { id: 'e265', source: 'redis', target: 'cache_system', edgeType: RELATIONSHIP_TYPES.IMPLEMENTS, label: '实现' },
    { id: 'e266', source: 'mysql', target: 'database_index', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e267', source: 'postgresql', target: 'database_index', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e268', source: 'database_index', target: 'binary_search_tree', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e269', source: 'database_index', target: 'hash_table', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },

    // ===== 软件工程实践关系 =====
    { id: 'e270', source: 'tech_lead', target: 'code_review', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '管理' },
    { id: 'e271', source: 'tech_lead', target: 'design_review', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '管理' },
    { id: 'e272', source: 'expert_engineer', target: 'technical_sharing', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e273', source: 'best_practices', target: 'code_review', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '执行' },
    { id: 'e274', source: 'communication', target: 'stakeholder', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e275', source: 'communication', target: 'product_manager', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e276', source: 'communication', target: 'business_analyst', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },

    // ===== 机器学习理论关系 =====
    { id: 'e277', source: 'bias_variance_tradeoff', target: 'overfitting', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e278', source: 'bias_variance_tradeoff', target: 'generalization', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '决定' },
    { id: 'e279', source: 'no_free_lunch', target: 'ml', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },

    // ===== 模型可解释性关系 =====
    { id: 'e280', source: 'model_interpretability', target: 'feature_importance', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e281', source: 'model_interpretability', target: 'feature_attribution', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e282', source: 'model_interpretability_methods', target: 'feature_importance', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e283', source: 'model_interpretability_methods', target: 'feature_attribution', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },

    // ===== 深度学习问题关系 =====
    { id: 'e284', source: 'gradient_vanishing', target: 'gradient_exploding', edgeType: RELATIONSHIP_TYPES.COMPARES_WITH, label: '对比' },
    { id: 'e285', source: 'transformer', target: 'gradient_vanishing', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '缓解' },
    { id: 'e286', source: 'rnn', target: 'gradient_vanishing', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '易引发' },

    // ===== 无监督学习细化关系 =====
    { id: 'e287', source: 'unsupervised_learning', target: 'dimensionality_reduction', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e288', source: 'unsupervised_learning', target: 'association_rule_learning', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },

    // ===== 其他补充关系 =====
    { id: 'e289', source: 'speech_recognition', target: 'rnn', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e290', source: 'speech_recognition', target: 'transformer', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },

    // ===== 神经网络基础关系 =====
    { id: 'e291', source: 'deep_learning', target: 'neural_network', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e292', source: 'neural_network', target: 'supervised_learning', edgeType: RELATIONSHIP_TYPES.IS_A, label: '属于' },
    { id: 'e293', source: 'cnn', target: 'neural_network', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e294', source: 'rnn', target: 'neural_network', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e295', source: 'neural_network', target: 'classification', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e296', source: 'neural_network', target: 'regression', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },

    // ===== 哈希相关补充关系 =====
    { id: 'e297', source: 'hash_collision', target: 'collision_resolution', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e298', source: 'collision_resolution', target: 'hash_function', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '优化' },

    // ===== 编程语言补充关系 =====
    { id: 'e299', source: 'cpp', target: 'dsa', edgeType: RELATIONSHIP_TYPES.IMPLEMENTS, label: '实现' },
    { id: 'e300', source: 'cpp', target: 'ml', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '支持' },
    { id: 'e301', source: 'java', target: 'dsa', edgeType: RELATIONSHIP_TYPES.IMPLEMENTS, label: '实现' },
    { id: 'e302', source: 'java', target: 'ml', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '支持' },
    { id: 'e303', source: 'tensorflow', target: 'cpp', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e304', source: 'tensorflow', target: 'python', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '支持' },
    { id: 'e305', source: 'pytorch', target: 'cpp', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e306', source: 'pytorch', target: 'python', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '支持' },
    { id: 'e307', source: 'senior_engineer', target: 'cpp', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },

    // ===== 数据库技术补充关系 =====
    { id: 'e308', source: 'mongodb', target: 'ml', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '支持' },
    { id: 'e309', source: 'mongodb', target: 'recsys', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e310', source: 'redis', target: 'recsys', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e311', source: 'mysql', target: 'ml', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '支持' },
    { id: 'e312', source: 'postgresql', target: 'ml', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '支持' },
    { id: 'e313', source: 'apache_spark', target: 'ml', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '支持' },
    { id: 'e314', source: 'apache_spark', target: 'data_preprocessing', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '支持' },

    // ===== 评估指标补充关系 =====
    { id: 'e315', source: 'roc_auc', target: 'classification', edgeType: RELATIONSHIP_TYPES.EVALUATES, label: '度量' },
    { id: 'e316', source: 'roc_auc', target: 'auc', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e317', source: 'resource_consumption', target: 'model_deployment', edgeType: RELATIONSHIP_TYPES.EVALUATES, label: '度量' },
    { id: 'e318', source: 'resource_consumption', target: 'complexity_analysis', edgeType: RELATIONSHIP_TYPES.IS_A, label: '属于' },

    // ===== 数据结构成本关系 =====
    { id: 'e319', source: 'array', target: 'insertion_cost', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '涉及' },
    { id: 'e320', source: 'array', target: 'deletion_cost', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '涉及' },
    { id: 'e321', source: 'linked_list', target: 'insertion_cost', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '优化' },
    { id: 'e322', source: 'linked_list', target: 'deletion_cost', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '优化' },
    { id: 'e323', source: 'insertion_cost', target: 'time_complexity', edgeType: RELATIONSHIP_TYPES.IS_A, label: '属于' },
    { id: 'e324', source: 'deletion_cost', target: 'time_complexity', edgeType: RELATIONSHIP_TYPES.IS_A, label: '属于' },
    { id: 'e325', source: 'dynamic_insertion', target: 'insertion_cost', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '决定' },
    { id: 'e326', source: 'dynamic_deletion', target: 'deletion_cost', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '决定' },

    // ===== 传统机器学习模型补充关系 =====
    { id: 'e327', source: 'linear_regression', target: 'regression', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '解决' },
    { id: 'e328', source: 'linear_regression', target: 'supervised_learning', edgeType: RELATIONSHIP_TYPES.IS_A, label: '属于' },
    { id: 'e329', source: 'logistic_regression', target: 'linear_regression', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '基于' },
    { id: 'e330', source: 'logistic_regression', target: 'supervised_learning', edgeType: RELATIONSHIP_TYPES.IS_A, label: '属于' },
    { id: 'e331', source: 'decision_tree', target: 'classification', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '解决' },
    { id: 'e332', source: 'decision_tree', target: 'regression', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '解决' },
    { id: 'e333', source: 'decision_tree', target: 'feature_importance', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '支持' },
    { id: 'e334', source: 'kmeans', target: 'unsupervised_learning', edgeType: RELATIONSHIP_TYPES.IS_A, label: '属于' },
    { id: 'e335', source: 'pca', target: 'unsupervised_learning', edgeType: RELATIONSHIP_TYPES.IS_A, label: '属于' },
    { id: 'e336', source: 'pca', target: 'feature_engineering', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '用于' },

    // ===== 数据结构与数学基础关系 =====
    { id: 'e337', source: 'dsa', target: 'complexity_analysis', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e338', source: 'dsa', target: 'linear_structure', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e339', source: 'dsa', target: 'nonlinear_structure', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e340', source: 'mathematics', target: 'complexity_analysis', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '支持' },
    { id: 'e341', source: 'mathematics', target: 'linear_programming', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '支持' },

    // ===== 算法范式补充关系 =====
    { id: 'e342', source: 'dsa', target: 'greedy_algorithm', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e343', source: 'dsa', target: 'divide_conquer', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e344', source: 'dsa', target: 'dynamic_programming', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e345', source: 'divide_conquer', target: 'greedy_algorithm', edgeType: RELATIONSHIP_TYPES.COMPARES_WITH, label: '对比' },
    { id: 'e346', source: 'dynamic_programming', target: 'greedy_algorithm', edgeType: RELATIONSHIP_TYPES.COMPARES_WITH, label: '对比' },
    { id: 'e347', source: 'dynamic_programming', target: 'divide_conquer', edgeType: RELATIONSHIP_TYPES.COMPARES_WITH, label: '对比' },

    // ===== 排序算法细化关系 =====
    { id: 'e348', source: 'dsa', target: 'comparison_sort', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e349', source: 'dsa', target: 'non_comparison_sort', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e350', source: 'comparison_sort', target: 'non_comparison_sort', edgeType: RELATIONSHIP_TYPES.COMPARES_WITH, label: '对比' },
    { id: 'e351', source: 'insertion_sort', target: 'bubble_sort', edgeType: RELATIONSHIP_TYPES.COMPARES_WITH, label: '对比' },
    { id: 'e352', source: 'selection_sort', target: 'insertion_sort', edgeType: RELATIONSHIP_TYPES.COMPARES_WITH, label: '对比' },
    { id: 'e353', source: 'counting_sort', target: 'radix_sort', edgeType: RELATIONSHIP_TYPES.COMPARES_WITH, label: '对比' },
    { id: 'e354', source: 'radix_sort', target: 'bucket_sort', edgeType: RELATIONSHIP_TYPES.COMPARES_WITH, label: '对比' },

    // ===== 模型生命周期补充关系 =====
    { id: 'e355', source: 'model_lifecycle', target: 'data_collection', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e356', source: 'model_lifecycle', target: 'data_preprocessing', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e357', source: 'model_lifecycle', target: 'model_deployment', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e358', source: 'model_lifecycle', target: 'model_monitoring', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e359', source: 'mlops', target: 'model_lifecycle', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '管理' },

    // ===== MLOps与容器化关系 =====
    { id: 'e360', source: 'mlops', target: 'docker', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e361', source: 'mlops', target: 'kubernetes', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e362', source: 'model_deployment', target: 'docker', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },
    { id: 'e363', source: 'model_deployment', target: 'kubernetes', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '使用' },

    // ===== 数据可视化与工具关系 =====
    { id: 'e364', source: 'data_visualization', target: 'data_preprocessing', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e365', source: 'data_visualization', target: 'model_evaluation', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e366', source: 'jupyter_notebook', target: 'ml', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e367', source: 'junior_engineer', target: 'jupyter_notebook', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },

    // ===== 硬件与性能关系 =====
    { id: 'e368', source: 'parallel_computing', target: 'apache_spark', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e369', source: 'parallel_computing', target: 'tensorflow', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e370', source: 'parallel_computing', target: 'pytorch', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e371', source: 'hardware_interaction', target: 'gpu', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e372', source: 'hardware_interaction', target: 'parallel_computing', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e373', source: 'random_access', target: 'hardware_interaction', edgeType: RELATIONSHIP_TYPES.IS_A, label: '属于' },

    // ===== 推荐系统模型补充关系 =====
    { id: 'e374', source: 'collaborative_filtering', target: 'recsys', edgeType: RELATIONSHIP_TYPES.IS_A, label: '属于' },
    { id: 'e375', source: 'content_based_filtering', target: 'recsys', edgeType: RELATIONSHIP_TYPES.IS_A, label: '属于' },
    { id: 'e376', source: 'wide_deep', target: 'recsys', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e377', source: 'deepfm', target: 'recsys', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e378', source: 'wide_deep', target: 'deepfm', edgeType: RELATIONSHIP_TYPES.COMPARES_WITH, label: '对比' },
    { id: 'e379', source: 'matrix_factorization', target: 'dimensionality_reduction', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },

    // ===== NLP模型补充关系 =====
    { id: 'e380', source: 'bert', target: 'nlp', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e381', source: 'gpt', target: 'nlp', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e382', source: 'bert', target: 'gpt', edgeType: RELATIONSHIP_TYPES.COMPARES_WITH, label: '对比' },
    { id: 'e383', source: 'bert', target: 'text_classification', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e384', source: 'bert', target: 'named_entity_recognition', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e385', source: 'gpt', target: 'natural_language_generation', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e386', source: 'word2vec', target: 'nlp', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e387', source: 'seq2seq', target: 'nlp', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },

    // ===== CV模型补充关系 =====
    { id: 'e388', source: 'cnn', target: 'image_classification', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e389', source: 'cnn', target: 'image_segmentation', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e390', source: 'pooling', target: 'dimensionality_reduction', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e391', source: 'filter', target: 'kernel', edgeType: RELATIONSHIP_TYPES.COMPARES_WITH, label: '对比' },

    // ===== 数据准备细化关系 =====
    { id: 'e392', source: 'data_collection', target: 'data_preparation', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '导向' },
    { id: 'e393', source: 'data_preparation', target: 'data_preprocessing', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e394', source: 'data_preparation', target: 'feature_engineering', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '导向' },
    { id: 'e395', source: 'training_set', target: 'model_training', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '用于' },
    { id: 'e396', source: 'validation_set', target: 'hyperparameter_tuning', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '用于' },
    { id: 'e397', source: 'test_set', target: 'model_evaluation', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '用于' },

    // ===== 文本处理细化关系 =====
    { id: 'e398', source: 'bag_of_words', target: 'text_classification', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e399', source: 'tfidf', target: 'text_classification', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e400', source: 'word2vec', target: 'contextual_embedding', edgeType: RELATIONSHIP_TYPES.COMPARES_WITH, label: '对比' },
    { id: 'e401', source: 'word_embedding', target: 'text_classification', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },

    // ===== 图算法应用补充 =====
    { id: 'e402', source: 'bfs', target: 'shortest_path', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '解决' },
    { id: 'e403', source: 'dfs', target: 'graph', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e404', source: 'dijkstra', target: 'shortest_path', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '解决' },
    { id: 'e405', source: 'prim', target: 'graph', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e406', source: 'kruskal', target: 'graph', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },

    // ===== 职业技能细化关系 =====
    { id: 'e407', source: 'junior_engineer', target: 'communication', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e408', source: 'senior_engineer', target: 'problem_solving', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e409', source: 'expert_engineer', target: 'strategic_thinking', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e410', source: 'tech_lead', target: 'leadership', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e411', source: 'senior_engineer', target: 'tensorflow', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e412', source: 'senior_engineer', target: 'pytorch', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e413', source: 'expert_engineer', target: 'operations_research', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },

    // ===== 软件工程实践补充 =====
    { id: 'e414', source: 'senior_engineer', target: 'code_review', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e415', source: 'expert_engineer', target: 'design_review', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '需要' },
    { id: 'e416', source: 'best_practices', target: 'design_review', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '执行' },
    { id: 'e417', source: 'best_practices', target: 'technical_sharing', edgeType: RELATIONSHIP_TYPES.CONTAINS, label: '包含' },
    { id: 'e418', source: 'best_practices', target: 'mlops', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },

    // ===== 优化算法应用补充 =====
    { id: 'e419', source: 'linear_programming', target: 'logistics_routing', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e420', source: 'integer_programming', target: 'resource_scheduling', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e421', source: 'genetic_algorithm', target: 'resource_scheduling', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },
    { id: 'e422', source: 'simulated_annealing', target: 'resource_scheduling', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用于' },

    // ===== 机器学习问题关系 =====
    { id: 'e423', source: 'overfitting', target: 'data_leakage', edgeType: RELATIONSHIP_TYPES.COMPARES_WITH, label: '对比' },
    { id: 'e424', source: 'overfitting', target: 'model_training', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '发生于' },
    { id: 'e425', source: 'data_leakage', target: 'data_preprocessing', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '发生于' },
    { id: 'e426', source: 'k_fold_cv', target: 'overfitting', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '防止' },
    { id: 'e427', source: 'gradient_exploding', target: 'deep_learning', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '发生于' },

    // ===== 复杂度与算法关系 =====
    { id: 'e428', source: 'bubble_sort', target: 'worst_case', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '存在' },
    { id: 'e429', source: 'quick_sort', target: 'average_case', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '优化' },
    { id: 'e430', source: 'merge_sort', target: 'best_case', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '维持' },
    { id: 'e431', source: 'binary_search', target: 'time_complexity', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '优化' },
    { id: 'e432', source: 'hash_table', target: 'time_complexity', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '优化' },

    // ===== 业务与技术关系 =====
    { id: 'e433', source: 'product_manager', target: 'problem_definition', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '负责' },
    { id: 'e434', source: 'business_analyst', target: 'data_collection', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '负责' },
    { id: 'e435', source: 'stakeholder', target: 'model_deployment', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '关注' },
    { id: 'e436', source: 'stakeholder', target: 'model_monitoring', edgeType: RELATIONSHIP_TYPES.DEPENDS_ON, label: '关注' },

    // ===== 缓存与数据库索引关系 =====
    { id: 'e437', source: 'cache_system', target: 'latency', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '优化' },
    { id: 'e438', source: 'cache_system', target: 'throughput', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '优化' },
    { id: 'e439', source: 'database_index', target: 'latency', edgeType: RELATIONSHIP_TYPES.AFFECTS, label: '优化' },
    { id: 'e440', source: 'database_index', target: 'binary_search', edgeType: RELATIONSHIP_TYPES.APPLIES_TO, label: '应用' },
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
