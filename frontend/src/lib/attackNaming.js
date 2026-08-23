export const ATTACK_LABELS = {
  fgsm: "FGSM (Fast Gradient)",
  pgd: "PGD (Projected Gradient)",
  mi_fgsm: "MI-FGSM (Momentum FGSM)",
  cw_l2: "C&W L2 (Carlini-Wagner)",
  tog: "TOG (Targeted BBox)",
  dpatch: "DPatch (Physical Patch)",
  depth_fog: "Sương mù (Fog)",
  depth_rain: "Mưa rơi (Rain)",
  object_occlusion: "Vật cản (Occlusion)",
  sensor_fault: "Lỗi cảm biến (Sensor)",
  gaussian_noise: "Nhiễu hạt Gaussian",
  motion_blur: "Chuyển động mờ (Blur)",
  defocus_blur: "Lấy nét mờ (Defocus)",
  contrast: "Độ tương phản (Contrast)",
  brightness: "Độ sáng (Brightness)",
  pixelate: "Pixelate (Vỡ hạt)",
  jpeg_compression: "Nén JPEG",
  square_attack: "Square (Blackbox)",
};

export const SHORT_ATTACK_LABELS = {
  fgsm: "FGSM",
  pgd: "PGD",
  mi_fgsm: "MI-FGSM",
  cw_l2: "C&W L2",
  tog: "TOG",
  dpatch: "DPatch",
  depth_fog: "Sương mù",
  depth_rain: "Mưa rơi",
  object_occlusion: "Vật cản",
  sensor_fault: "Lỗi cảm biến",
  gaussian_noise: "Nhiễu Gaussian",
  motion_blur: "Mờ chuyển động",
  defocus_blur: "Mờ lấy nét",
  contrast: "Độ tương phản",
  brightness: "Độ sáng",
  pixelate: "Pixelate",
  jpeg_compression: "Nén JPEG",
  square_attack: "Square Attack",
};

/**
 * Normalizes dataset strings like "kitti_val", "KITTI", "kitti" to a canonical key like "kitti".
 */
export function normalizeDatasetKey(name) {
  if (!name) return "default";
  const str = String(name).toLowerCase();
  if (str.includes("kitti")) return "kitti";
  if (str.includes("coco")) return "coco";
  if (str.includes("bdd")) return "bdd100k";
  if (str.includes("shape") || str.includes("synthetic")) return "synthetic_shapes";
  if (str.includes("folder")) return "folder_dataset";
  return str.replace(/[^a-z0-9]/g, "");
}

/**
 * Extracts a normalized, canonical key for an attack cell/sample to guarantee exact deduplication.
 * E.g. "contrast_3" whether the raw attack was "recipe-bb804a..." or "contrast".
 */
export function getCanonicalAttackKey(cellOrAttack, explicitSeverity = null, report = null) {
  if (!cellOrAttack) return "attack_unknown";

  const isObject = typeof cellOrAttack === "object" && cellOrAttack !== null;
  const rawAttack = isObject
    ? cellOrAttack.attack || cellOrAttack.attack_name || cellOrAttack.name || ""
    : String(cellOrAttack);
  const severity = isObject ? (cellOrAttack.severity ?? explicitSeverity) : explicitSeverity;
  const isRecipeHash = /^recipe-[a-f0-9]+$/i.test(rawAttack) || rawAttack.startsWith("recipe-");

  let steps = isObject ? (cellOrAttack.recipe_steps || cellOrAttack.metrics?.recipe_steps || null) : null;

  if (!steps && report) {
    const candidateList = [...(report.sample_results || []), ...(report.worst_cases || [])];
    const matchingSample = candidateList.find((s) => {
      const matchAttack = s.attack === rawAttack || isRecipeHash;
      const matchSev = severity != null ? s.severity === severity : true;
      return matchAttack && matchSev && Array.isArray(s.recipe_steps) && s.recipe_steps.length > 0;
    });
    if (matchingSample) {
      steps = matchingSample.recipe_steps;
    }
  }

  if (!steps && isRecipeHash && report?.provenance?.recipe?.steps?.length) {
    steps = report.provenance.recipe.steps;
  }

  if (Array.isArray(steps) && steps.length > 0) {
    const stepSignatures = steps.map((step) => {
      const name = step.attack_name || step.attack || step.name || "attack";
      const sev = step.severity != null ? step.severity : (severity || 1);
      return `${name}_${sev}`;
    });
    return stepSignatures.join("+");
  }

  const cleanRaw = rawAttack.replace(/^recipe-[a-f0-9]+$/i, "recipe").trim();
  return `${cleanRaw}_${severity || 1}`;
}

/**
 * Resolves a human-readable, distinguishable attack name from a cell or sample object.
 * Removes ugly hashes like `recipe-bb804a059aa28ad2d0e5` and replaces them with actual step names.
 */
export function getDescriptiveAttackName(cellOrAttack, explicitSeverity = null, report = null) {
  if (!cellOrAttack) return "Tấn công";

  const isObject = typeof cellOrAttack === "object" && cellOrAttack !== null;
  const rawAttack = isObject
    ? cellOrAttack.attack || cellOrAttack.attack_name || cellOrAttack.name || ""
    : String(cellOrAttack);
  const severity = isObject ? (cellOrAttack.severity ?? explicitSeverity) : explicitSeverity;
  const isRecipeHash = /^recipe-[a-f0-9]+$/i.test(rawAttack) || rawAttack.startsWith("recipe-");

  // 1. Try to find recipe steps from the object itself
  let steps = isObject ? (cellOrAttack.recipe_steps || cellOrAttack.metrics?.recipe_steps || null) : null;

  // 2. If not found and report is provided, check sample_results or worst_cases
  if (!steps && report) {
    const candidateList = [...(report.sample_results || []), ...(report.worst_cases || [])];
    const matchingSample = candidateList.find((s) => {
      const matchAttack = s.attack === rawAttack || isRecipeHash;
      const matchSev = severity != null ? s.severity === severity : true;
      return matchAttack && matchSev && Array.isArray(s.recipe_steps) && s.recipe_steps.length > 0;
    });
    if (matchingSample) {
      steps = matchingSample.recipe_steps;
    }
  }

  // 3. If still not found and is recipe hash, check report.provenance.recipe.steps
  if (!steps && isRecipeHash && report?.provenance?.recipe?.steps?.length) {
    steps = report.provenance.recipe.steps;
  }

  // 4. Format if steps are found
  if (Array.isArray(steps) && steps.length > 0) {
    const formattedParts = steps.map((step) => {
      const stepRaw = step.attack_name || step.attack || step.name || "attack";
      const stepLabel = SHORT_ATTACK_LABELS[stepRaw] || ATTACK_LABELS[stepRaw] || stepRaw.replace(/_/g, " ").toUpperCase();
      return stepLabel;
    });
    const joined = formattedParts.join(" + ");
    const sevStr = severity != null ? ` (Cấp ${severity})` : "";
    return `${joined}${sevStr}`;
  }

  // 5. If not a hash, format directly
  if (rawAttack && !isRecipeHash) {
    const label = SHORT_ATTACK_LABELS[rawAttack] || ATTACK_LABELS[rawAttack] || rawAttack.replace(/_/g, " ").toUpperCase();
    const sevStr = severity != null ? ` (Cấp ${severity})` : "";
    return `${label}${sevStr}`;
  }

  // 6. Fallback if it is an unknown recipe hash
  const sevStr = severity != null ? ` (Cấp ${severity})` : "";
  return `Tổ hợp đòn đánh${sevStr}`;
}
