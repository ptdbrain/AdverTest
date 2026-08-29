import { Suspense } from "react";

import ConfigureAttackPage from "../experiments/[id]/attack/page";

export default function AttackAliasPage() {
  return (
    <Suspense fallback={null}>
      <ConfigureAttackPage />
    </Suspense>
  );
}
