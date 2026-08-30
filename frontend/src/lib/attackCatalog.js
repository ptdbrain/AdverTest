/** Return only attacks that the server declares compatible with the task. */
export function visibleAttacks(catalog, taskId) {
  return catalog.filter((attack) => {
    const compatibleTasks = attack.task_compatibility || attack.task_ids || attack.required_tasks || [];
    return Array.isArray(compatibleTasks) && compatibleTasks.includes(taskId);
  });
}
