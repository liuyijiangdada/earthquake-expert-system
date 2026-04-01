export const WELCOME =
  '你好，我是地震知识助手。你可以问我关于成因、避险、预警、历史震例等问题；我也会尽量结合后台知识图谱中的地震记录来回答。'

export const QUICK_GROUPS = [
  {
    title: '科普与概念',
    ariaLabel: '科普与概念',
    items: [
      { icon: 'fa-mountain', label: '历史著名地震', q: '历史上发生过哪些较为著名的地震事件？请简要列举并说明。' },
      { icon: 'fa-ruler-vertical', label: '震级与烈度', q: '震级和烈度有什么区别？' },
      { icon: 'fa-bell', label: '预警与预报', q: '地震预警和地震预报有什么不同？' },
      { icon: 'fa-water', label: '震感差异', q: '为什么有的地方震感强、有的地方弱？' },
    ],
  },
  {
    title: '应急与安全',
    ariaLabel: '应急与安全',
    items: [
      { icon: 'fa-house-chimney-crack', label: '室内避险', q: '室内遇到地震时，正确的避险步骤是什么？' },
      { icon: 'fa-person-walking', label: '户外注意', q: '地震发生后，户外人员应注意哪些事项？' },
      { icon: 'fa-kit-medical', label: '应急包', q: '家庭应急包建议准备哪些物品？' },
    ],
  },
  {
    title: '结合知识图谱',
    ariaLabel: '结合知识图谱',
    items: [
      {
        icon: 'fa-database',
        label: '高震级事件',
        q: '请根据知识图谱信息，列举震级大于6级的地震事件（若有）。',
      },
    ],
  },
]
