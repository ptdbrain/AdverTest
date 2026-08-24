"use client";

import React, { useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import {
  Crosshair,
  Search,
  Sliders,
  Play,
  CheckCircle2,
  Trash2,
  GripVertical,
  Plus,
  RefreshCw,
  Sparkles,
  ArrowRight,
  ShieldAlert,
  SlidersHorizontal,
  ChevronDown,
  CloudRain,
  CloudFog,
  Snowflake,
  Sun,
  Camera,
  Layers,
} from "lucide-react";
import PageHeader from "@/components/layout/PageHeader";
import Card from "@/components/common/Card";
import Badge from "@/components/common/Badge";
import Button from "@/components/common/Button";
import { ATTACK_CATEGORIES } from "@/data/mockData";
import { cn } from "@/lib/utils";

export default function ConfigureAttackPage() {
  const params = useParams();
  const router = useRouter();
  const expId = params?.id || "EXP-2025-0512-001";

  const [mode, setMode] = useState("single"); // 'single' | 'combo'
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedAttackId, setSelectedAttackId] = useState("depth_fog");
  const [targetMode, setTargetMode] = useState("untargeted"); // 'untargeted' | 'targeted'
  const [lossFunction, setLossFunction] = useState("Cross Entropy");
  const [normType, setNormType] = useState("Linf");
  const [strength, setStrength] = useState(0.70);
  const [epsilonNum, setEpsilonNum] = useState(8);
  const [severityLevel, setSeverityLevel] = useState(3);
  const [stepSize, setStepSize] = useState("2/255");
  const [iterations, setIterations] = useState(20);
  const [randomInit, setRandomInit] = useState(true);
  const [clipImage, setClipImage] = useState(true);
  const [randomSeed, setRandomSeed] = useState(42);
  const [scope, setScope] = useState("full"); // 'full' | 'roi'
  const [patchSize, setPatchSize] = useState("64 × 64");
  const [patchPos, setPatchPos] = useState("bottom_right");
  const [depthAware, setDepthAware] = useState(true);
  const [isExecuting, setIsExecuting] = useState(false);

  // Attack Queue for combo
  const [attackQueue, setAttackQueue] = useState([
    { id: "depth_fog", name: "Depth Fog (Severity 3)", eps: "Sev-3", alpha: "Depth", iters: 1 },
    { id: "pgd", name: "PGD (ε=8/255)", eps: "8/255", alpha: "2/255", iters: 20 },
  ]);

  const allAttacksList = ATTACK_CATEGORIES.flatMap((c) => c.attacks);
  const currentAttack = allAttacksList.find((a) => a.id === selectedAttackId) || allAttacksList[0];
  const isWeatherAttack = currentAttack?.isWeather || selectedAttackId.startsWith("depth_") || selectedAttackId.includes("noise") || selectedAttackId.includes("blur") || selectedAttackId.includes("frost");

  const handleAddCombo = (attack) => {
    if (attackQueue.some((a) => a.id === attack.id)) return;
    setAttackQueue([...attackQueue, { id: attack.id, name: attack.name, eps: attack.defaultEps, alpha: "Auto", iters: 20 }]);
  };

  const handleRemoveQueue = (id) => {
    if (attackQueue.length <= 1) return;
    setAttackQueue(attackQueue.filter((a) => a.id !== id));
  };

  const handleStartAttack = () => {
    setIsExecuting(true);
    setTimeout(() => {
      router.push(`/experiments/${expId}/results`);
    }, 1200);
  };

  const filteredCategories = ATTACK_CATEGORIES.map((cat) => ({
    ...cat,
    attacks: cat.attacks.filter(
      (a) =>
        a.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        a.fullName.toLowerCase().includes(searchQuery.toLowerCase())
    ),
  })).filter((cat) => cat.attacks.length > 0);

  return (
    <div className="space-y-5 animate-fade-in">
      <PageHeader
        title="Cấu hình tấn công & Nhiễu môi trường đối kháng"
        subtitle="Chọn phương pháp tấn công gradient (PGD, FGSM), nhiễu vật lý (DPatch), hoặc nhiễu thời tiết thực tế (Depth Fog, Rain, Snow, Glare)."
        breadcrumb={[
          { label: "Trang chủ", href: "/dashboard" },
          { label: "Cấu hình bài toán", href: "/experiments/new" },
          { label: "Cấu hình tấn công" },
        ]}
        actions={
          <div className="flex items-center bg-white p-1 rounded-lg border border-slate-200 shadow-sm">
            <button
              type="button"
              onClick={() => setMode("single")}
              className={cn(
                "px-3 py-1 text-xs font-semibold rounded-md transition-colors",
                mode === "single" ? "bg-blue-600 text-white" : "text-slate-600 hover:text-slate-900"
              )}
            >
              Tấn công đơn
            </button>
            <button
              type="button"
              onClick={() => setMode("combo")}
              className={cn(
                "px-3 py-1 text-xs font-semibold rounded-md transition-colors",
                mode === "combo" ? "bg-blue-600 text-white" : "text-slate-600 hover:text-slate-900"
              )}
            >
              Tổ hợp tấn công (Combo)
            </button>
          </div>
        }
      />

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-5 items-start">
        {/* LEFT: Attack Library (4 Columns) */}
        <div className="xl:col-span-4 space-y-4">
          <Card
            title="Thư viện tấn công & Nhiễu tự nhiên"
            subtitle="7 nhóm phương pháp (Thời tiết, Gradient, Black-box, Patch, 3D...)"
          >
            {/* Search Input */}
            <div className="relative mb-3">
              <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-slate-400" />
              <input
                type="text"
                placeholder="Tìm kiếm phương pháp hoặc thời tiết (Fog, Rain, PGD...)"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-8 pr-3 py-1.5 text-xs rounded-lg border border-slate-200 bg-slate-50 focus:bg-white focus:outline-none focus:border-blue-500"
              />
            </div>

            {/* Categories & Algorithm Tiles */}
            <div className="space-y-3.5 max-h-[660px] overflow-y-auto pr-1">
              {filteredCategories.map((cat) => (
                <div key={cat.category} className="space-y-1.5">
                  <div className="text-[11px] font-bold text-slate-700 bg-slate-100 py-1 px-2 rounded flex items-center justify-between">
                    <span>{cat.category}</span>
                    <span className="text-[10px] text-slate-400 font-mono">({cat.attacks.length})</span>
                  </div>
                  <div className="grid grid-cols-1 gap-1.5">
                    {cat.attacks.map((atk) => {
                      const isSelected = selectedAttackId === atk.id;
                      return (
                        <div
                          key={atk.id}
                          onClick={() => {
                            setSelectedAttackId(atk.id);
                            if (mode === "combo") handleAddCombo(atk);
                          }}
                          className={cn(
                            "p-2 rounded-lg border text-left cursor-pointer transition-all flex items-center justify-between",
                            isSelected
                              ? "border-blue-600 bg-blue-50/70 shadow-sm ring-1 ring-blue-500/20"
                              : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50"
                          )}
                        >
                          <div>
                            <div className="text-xs font-bold text-slate-800 leading-tight flex items-center gap-1.5">
                              {atk.isWeather ? <span>🌦️</span> : <span>🎯</span>}
                              <span>{atk.name}</span>
                            </div>
                            <div className="text-[10px] text-slate-500 leading-tight mt-0.5">
                              {atk.fullName}
                            </div>
                          </div>
                          <Badge
                            variant={atk.isWeather ? "teal" : atk.norm === "Linf" ? "purple" : "primary"}
                            className="text-[10px] font-mono"
                          >
                            {atk.norm}
                          </Badge>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* CENTER: Attack Configuration (4 Columns) */}
        <div className="xl:col-span-4 space-y-4">
          {/* Section 1: Thuật toán & Mục tiêu */}
          <Card title="1. Phương pháp & Mục tiêu tác động">
            <div className="space-y-3 text-xs">
              <div>
                <label className="text-slate-500 font-medium block mb-1">Phương pháp đã chọn</label>
                <div className="p-2 rounded border border-slate-300 bg-slate-50 font-bold text-blue-700 font-mono flex items-center justify-between">
                  <span>{currentAttack.name}</span>
                  <Badge variant={isWeatherAttack ? "teal" : "primary"}>
                    {isWeatherAttack ? "Môi trường thực tế" : "Adversarial"}
                  </Badge>
                </div>
                <p className="text-[11px] text-slate-500 mt-1 italic">
                  {currentAttack.fullName}
                </p>
              </div>

              {!isWeatherAttack && (
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-slate-500 font-medium block mb-1">Hàm mất mát (Loss)</label>
                    <select
                      value={lossFunction}
                      onChange={(e) => setLossFunction(e.target.value)}
                      className="w-full p-2 rounded border border-slate-300 bg-white"
                    >
                      <option value="Cross Entropy">Cross Entropy</option>
                      <option value="CW Loss">CW Margin Loss</option>
                      <option value="Focal Loss">Focal Loss</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-slate-500 font-medium block mb-1">Chuẩn ràng buộc (Norm)</label>
                    <div className="flex gap-1.5">
                      {["Linf", "L2", "L1"].map((n) => (
                        <button
                          key={n}
                          type="button"
                          onClick={() => setNormType(n)}
                          className={cn(
                            "flex-1 py-1.5 rounded border text-xs font-semibold transition-colors",
                            normType === n
                              ? "bg-blue-600 text-white border-blue-600"
                              : "bg-white text-slate-700 border-slate-300 hover:bg-slate-50"
                          )}
                        >
                          {n === "Linf" ? "L∞" : n}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </Card>

          {/* Section 2: Tham số tấn công / Cường độ thời tiết */}
          <Card title="2. Thiết lập mức độ & Tham số chi tiết">
            <div className="space-y-4 text-xs">
              {isWeatherAttack ? (
                /* Weather & Corruption Controls */
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between font-medium mb-1">
                      <span className="text-slate-700 font-semibold">Cấp độ khắc nghiệt (Severity Level)</span>
                      <span className="font-bold text-teal-700 bg-teal-50 px-2 py-0.5 rounded border border-teal-200">
                        Cấp {severityLevel} / 5 ({severityLevel === 1 ? "Nhẹ" : severityLevel === 2 ? "Vừa" : severityLevel === 3 ? "Nặng" : severityLevel === 4 ? "Rất nặng" : "Cực đoan"})
                      </span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="5"
                      step="1"
                      value={severityLevel}
                      onChange={(e) => setSeverityLevel(parseInt(e.target.value))}
                      className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-teal-600"
                    />
                    <div className="flex justify-between text-[10px] text-slate-500 mt-1 font-mono">
                      <span>1 (Sương/Mưa nhẹ)</span>
                      <span>3 (Trung bình)</span>
                      <span>5 (Bão/Tuyết dày)</span>
                    </div>
                  </div>

                  <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200 space-y-2">
                    <label className="flex items-center justify-between cursor-pointer">
                      <span className="text-slate-700 font-medium">Mô phỏng chiều sâu quang học (Depth-Aware)</span>
                      <input
                        type="checkbox"
                        checked={depthAware}
                        onChange={(e) => setDepthAware(e.target.checked)}
                        className="rounded text-teal-600"
                      />
                    </label>
                    <p className="text-[10px] text-slate-500">
                      Tự động tính ma trận Depth Map từ stereo/monocular camera để hạt sương/tuyết/mưa dày hơn ở các đối tượng ở xa.
                    </p>
                  </div>
                </div>
              ) : (
                /* Adversarial Perturbation Controls */
                <div className="space-y-3">
                  {/* Strength slider */}
                  <div>
                    <div className="flex justify-between font-medium mb-1">
                      <span className="text-slate-700">Mức độ tấn công (Strength)</span>
                      <span className="font-bold text-blue-600">{strength.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.0"
                      max="1.0"
                      step="0.05"
                      value={strength}
                      onChange={(e) => setStrength(parseFloat(e.target.value))}
                      className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                    />
                  </div>

                  {/* Epsilon slider */}
                  <div>
                    <div className="flex justify-between font-medium mb-1">
                      <span className="text-slate-700">Ngưỡng nhiễu Epsilon (ε)</span>
                      <span className="font-bold font-mono text-red-600">{epsilonNum}/255 ({ (epsilonNum/255).toFixed(4) })</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="16"
                      step="1"
                      value={epsilonNum}
                      onChange={(e) => setEpsilonNum(parseInt(e.target.value))}
                      className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-red-600"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="text-slate-500 font-medium block mb-1">Bước nhảy (Alpha)</label>
                      <input
                        type="text"
                        value={stepSize}
                        onChange={(e) => setStepSize(e.target.value)}
                        className="w-full px-2.5 py-1.5 rounded border border-slate-300 font-mono text-slate-800"
                      />
                    </div>
                    <div>
                      <label className="text-slate-500 font-medium block mb-1">Số vòng lặp (Iters)</label>
                      <input
                        type="number"
                        value={iterations}
                        onChange={(e) => setIterations(parseInt(e.target.value) || 1)}
                        className="w-full px-2.5 py-1.5 rounded border border-slate-300 font-mono text-slate-800"
                      />
                    </div>
                  </div>
                </div>
              )}

              <div className="flex items-center justify-between pt-2 border-t border-slate-100">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={randomInit}
                    onChange={(e) => setRandomInit(e.target.checked)}
                    className="rounded text-blue-600"
                  />
                  <span className="text-slate-700 font-medium">Khởi tạo ngẫu nhiên (Random Init)</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={clipImage}
                    onChange={(e) => setClipImage(e.target.checked)}
                    className="rounded text-blue-600"
                  />
                  <span className="text-slate-700 font-medium">Cắt ngưỡng ảnh [0, 1]</span>
                </label>
              </div>
            </div>
          </Card>

          {/* Section 3: Ràng buộc & Cài đặt Patch */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card title="3. Ràng buộc vùng">
              <div className="space-y-2 text-xs">
                <div>
                  <label className="text-slate-500 font-medium block mb-1">Phạm vi áp dụng</label>
                  <select
                    value={scope}
                    onChange={(e) => setScope(e.target.value)}
                    className="w-full p-1.5 rounded border border-slate-300 bg-white"
                  >
                    <option value="full">Toàn bộ ảnh (Full scene)</option>
                    <option value="roi">Vùng ROI xe cộ / người đi bộ</option>
                  </select>
                </div>
              </div>
            </Card>

            <Card title="4. Physical Patch (Nếu có)">
              <div className="space-y-2 text-xs">
                <div>
                  <label className="text-slate-500 font-medium block mb-1">Kích thước Patch</label>
                  <select
                    value={patchSize}
                    onChange={(e) => setPatchSize(e.target.value)}
                    className="w-full p-1.5 rounded border border-slate-300 bg-white"
                  >
                    <option value="64 × 64">64 × 64 px (Chuẩn)</option>
                    <option value="128 × 128">128 × 128 px (Lớn)</option>
                  </select>
                </div>
              </div>
            </Card>
          </div>
        </div>

        {/* RIGHT: Attack Summary, Queue & Live Preview (4 Columns) */}
        <div className="xl:col-span-4 space-y-4 sticky top-[74px]">
          {/* Attack Summary */}
          <Card title="Tóm tắt cấu hình tác động">
            <div className="space-y-1.5 text-xs divide-y divide-slate-100 font-medium mb-3">
              <div className="flex justify-between py-1">
                <span className="text-slate-500">Thuật toán / Nhiễu</span>
                <span className="font-bold text-blue-600">{currentAttack.name}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-500">Phân loại</span>
                <span className="text-slate-800">{isWeatherAttack ? "Thời tiết tự nhiên" : "Đối kháng nhân tạo"}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-500">Mức độ / Cường độ</span>
                <span className="font-mono text-red-600 font-bold">
                  {isWeatherAttack ? `Severity ${severityLevel} / 5` : `L∞ (ε = ${epsilonNum}/255)`}
                </span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-500">Vùng tác động</span>
                <span className="text-slate-800">{scope === "full" ? "Toàn bộ khung hình" : "Vùng ROI"}</span>
              </div>
            </div>

            {/* Attack Queue */}
            <div className="mb-3">
              <div className="text-[11px] font-bold text-slate-500 mb-1.5">
                Thứ tự chuỗi tác động (Queue)
              </div>
              <div className="space-y-1.5">
                {attackQueue.map((item, idx) => (
                  <div
                    key={item.id}
                    className="p-2 rounded bg-slate-50 border border-slate-200 flex items-center justify-between text-xs font-mono"
                  >
                    <div className="flex items-center gap-2 truncate">
                      <GripVertical className="w-3.5 h-3.5 text-slate-400 flex-shrink-0" />
                      <span className="font-bold text-blue-600 truncate">{idx + 1}. {item.name}</span>
                    </div>
                    {attackQueue.length > 1 && (
                      <button
                        type="button"
                        onClick={() => handleRemoveQueue(item.id)}
                        className="text-red-500 hover:text-red-700 ml-2"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Live Perturbation Preview */}
            <div className="mb-4">
              <div className="text-[11px] font-bold text-slate-500 mb-1.5">
                Xem trước tác động (Live Preview)
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="h-24 rounded bg-slate-200 border border-slate-300 flex flex-col items-center justify-center text-slate-600 text-[10px] font-medium text-center p-1">
                  <span>Ảnh KITTI gốc</span>
                  <span className="text-[9px] text-slate-400 font-mono">1242 × 375</span>
                </div>
                <div className="h-24 rounded bg-slate-300 border border-blue-400 flex flex-col items-center justify-center text-blue-900 text-[10px] font-medium text-center p-1 relative overflow-hidden bg-gradient-to-b from-slate-200 to-slate-400">
                  <span>{currentAttack.name}</span>
                  <span className="text-[9px] text-blue-700 font-bold">
                    {isWeatherAttack ? `Cấp độ ${severityLevel}` : `ε = ${epsilonNum}/255`}
                  </span>
                  <span className="absolute bottom-1 right-1 px-1 rounded bg-blue-600 text-white text-[8px] font-bold">
                    SIMULATED
                  </span>
                </div>
              </div>
              <div className="mt-1.5 text-[10px] text-emerald-600 font-medium">
                ✓ Sẵn sàng suy luận đánh giá trên mô hình đã cấu hình
              </div>
            </div>

            {/* CTA Button */}
            <Button
              variant="primary"
              size="lg"
              className="w-full text-sm font-bold bg-blue-600 hover:bg-blue-700 py-3 shadow-md"
              icon={isExecuting ? RefreshCw : Play}
              disabled={isExecuting}
              onClick={handleStartAttack}
            >
              {isExecuting ? "Đang chạy suy luận & tác động..." : "▶ Bắt đầu tấn công / Tác động"}
            </Button>
          </Card>
        </div>
      </div>
    </div>
  );
}
