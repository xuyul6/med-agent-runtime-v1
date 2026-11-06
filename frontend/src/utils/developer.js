// The backend decides the mode: /health reports developer_tools, and developer routes answer 404 when it is off.
export const USER_VIEWS = [['chat', '咨询'], ['appointments', '预约']]
export const DEVELOPER_VIEWS = [['knowledge', '参考资料'], ['roles', '角色契约'], ['evaluation', '评测']]

export function developerMode(health) {
  return health?.developer_tools === true
}

export function workspaceViews(developer) {
  return developer ? [...USER_VIEWS, ...DEVELOPER_VIEWS] : USER_VIEWS
}

export function visibleView(view, developer) {
  return workspaceViews(developer).some(([key]) => key === view) ? view : 'chat'
}

export function connectionLabel(health, developer) {
  if (!health) return '未连接'
  if (health.preview) return '离线预览'
  if (developer) return health.status === 'ok' ? '在线模型' : '依赖降级'
  return health.status === 'ok' ? '服务正常' : '部分服务不可用'
}

export function degradedNotice(developer) {
  return developer ? '本次存在运行降级，请查看执行详情。' : '部分内容未能生成，可以稍后再试。'
}
