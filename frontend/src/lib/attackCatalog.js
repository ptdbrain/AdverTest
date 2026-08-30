/** Return only attacks that the server declares compatible with the task. */
export function visibleAttacks(catalog, taskId) {
  return catalog.filter((attack) => Array.isArray(attack.task_compatibility) && attack.task_compatibility.includes(taskId));
}
