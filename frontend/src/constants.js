export const WELCOME =
  '你好，我是地震应急智能助手。系统会按「震前 · 震中 · 震后」理解你的问题，并协同知识图谱、向量检索与实时震情数据作答。下方可点选场景化快捷提问，也可上传图片进行多模态分析。'

/** 与论文三阶段 + 后端 phase_classifier 对齐 */
export const QUICK_GROUPS = [
  {
    title: '震前 · 防御准备',
    phase: '震前',
    ariaLabel: '震前防御准备',
    items: [
      { icon: 'fa-kit-medical', label: '家庭应急包', q: '家庭应急包该准备什么？请列清单。' },
      { icon: 'fa-school', label: '学校演练', q: '学校如何组织防震演练？' },
      { icon: 'fa-mobile-screen', label: '地震预警', q: '如何开启手机地震预警功能？' },
      { icon: 'fa-house-chimney', label: '居家防震', q: '居家防震需要做哪些检查和准备？' },
    ],
  },
  {
    title: '震中 · 应急响应',
    phase: '震中',
    ariaLabel: '震中应急响应',
    items: [
      { icon: 'fa-house-chimney-crack', label: '室内避险', q: '地震发生时室内正确的避险步骤是什么？' },
      { icon: 'fa-person-walking', label: '室外避险', q: '室外开阔地带如何避险？' },
      { icon: 'fa-map-location-dot', label: '最近避难所', q: '我在成都，最近的应急避难所在哪？' },
      { icon: 'fa-satellite-dish', label: '最新震情', q: '刚才地震多大？震中在哪？' },
    ],
  },
  {
    title: '震后 · 恢复重建',
    phase: '震后',
    ariaLabel: '震后恢复重建',
    items: [
      { icon: 'fa-building-circle-check', label: '房屋安全', q: '我家房子震后安全吗？怎么判断？' },
      { icon: 'fa-file-contract', label: '救助政策', q: '政府有哪些灾后重建补贴政策？' },
      { icon: 'fa-cloud-bolt', label: '余震风险', q: '现在还有余震风险吗？' },
      { icon: 'fa-database', label: '高震级记录', q: '请根据知识图谱列举震级大于6级的地震事件（若有）。' },
    ],
  },
]

export const PHASE_META = {
  震前: { class: 'phase-pre', icon: 'fa-shield-halved', hint: '科普 · 预防' },
  震中: { class: 'phase-during', icon: 'fa-bolt', hint: '实时 · 避险' },
  震后: { class: 'phase-post', icon: 'fa-hands-holding-circle', hint: '恢复 · 政策' },
  通用: { class: 'phase-general', icon: 'fa-circle-info', hint: '综合问答' },
}
