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
  const [selectedAttackId, setSelectedAttackId] = useState("pgd");
  const [targetMode, setTargetMode] = useState("untargeted"); // 'untargeted' | 'targeted'
  const [lossFunction, setLossFunction] = useState("Cross Entropy");
  const [normType, setNormType] = useState("Linf");
  const [strength, setStrength] = useState(0.70);
  const [epsilonNum, setEpsilonNum] = useState(8);
  const [stepSize, setStepSize] = useState("2/255");
  const [iterations, setIterations] = useState(20);
  const [randomInit, setRandomInit] = useState(true);
  const [clipImage, setClipImage] = useState(true);
  const [randomSeed, setRandomSeed] = useState(42);
  const [scope, setScope] = useState("full"); // 'full' | 'roi'
  const [patchSize, setPatchSize] = useState("64 × 64");
  const [patchPos, setPatchPos] = useState("bottom_right");
  const [isExecuting, setIsExecuting] = useState(false);

  // Attack Queue for combo
  const [attackQueue, setAttackQueue] = useState([
    { id: "pgd", name: "PGD", eps: "8/255", alpha: "2/255", iters: 20 },
  ]);

  const handleAddCombo = (attack) => {
    if (attackQueue.some((a) => a.id === attack.id)) return;
    setAttackQueue([...attackQueue, { id: attack.id, name: attack.name, eps: "8/255", alpha: "2/255", iters: 20 }]);
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
        title="Cấu hình tấn công đối kháng"
        subtitle="Chọn thuật toán, thiết lập tham số và cấu hình cách thức tấn công mô hình"
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
        {/* LEFT: Attack Library (3 Columns) */}
        <div className="xl:col-span-3 space-y-4">
          <Card title="Thư viện tấn công" subtitle="48 phương pháp tấn công đối kháng">
            {/* Search Input */}
            <div className="relative mb-3">
              <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-slate-400" />
              <input
                type="text"
                placeholder="Tìm kiếm thuật toán tấn công..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-8 pr-3 py-1.5 text-xs rounded-lg border border-slate-200 bg-slate-50 focus:bg-white focus:outline-none focus:border-blue-500"
              />
            </div>

            {/* Categories & Algorithm Tiles */}
            <div className="space-y-3.5 max-h-[640px] overflow-y-auto pr-1">
              {filteredCategories.map((cat) => (
                <div key={cat.category} className="space-y-1.5">
                  <div className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
                    {cat.category}
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
                              ? "border-blue-600 bg-blue-50/70 shadow-sm"
                              : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50"
                          )}
                        >
                          <div>
                            <div className="text-xs font-bold text-slate-800 leading-tight">
                              {atk.name}
                            </div>
                            <div className="text-[10px] text-slate-500 leading-tight mt-0.5">
                              {atk.fullName}
                            </div>
                          </div>
                          <Badge variant="purple" className="text-[10px] font-mono">
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

        {/* CENTER: Attack Configuration (5 Columns) */}
        <div className="xl:col-span-5 space-y-4">
          {/* Section 1: Thuật toán & Mục tiêu */}
          <Card title="1. Thuật toán & mục tiêu tấn công">
            <div className="space-y-3 text-xs">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-slate-500 font-medium block mb-1">Thuật toán đã chọn</label>
                  <div className="p-2 rounded border border-slate-300 bg-slate-50 font-bold text-blue-700 font-mono">
                    {selectedAttackId.toUpperCase()} (Projected Gradient Descent)
                  </div>
                </div>
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
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-slate-500 font-medium block mb-1">Chuẩn ràng buộc (Norm)</label>
                  <div className="flex gap-2">
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
                <div>
                  <label className="text-slate-500 font-medium block mb-1">Chế độ mục tiêu</label>
                  <div className="flex gap-3 pt-1">
                    <label className="flex items-center gap-1.5 cursor-pointer">
                      <input
                        type="radio"
                        name="target_mode"
                        checked={targetMode === "untargeted"}
                        onChange={() => setTargetMode("untargeted")}
                        className="text-blue-600"
                      />
                      <span>Không mục tiêu</span>
                    </label>
                    <label className="flex items-center gap-1.5 cursor-pointer">
                      <input
                        type="radio"
                        name="target_mode"
                        checked={targetMode === "targeted"}
                        onChange={() => setTargetMode("targeted")}
                        className="text-blue-600"
                      />
                      <span>Có mục tiêu</span>
                    </label>
                  </div>
                </div>
              </div>
            </div>
          </Card>

          {/* Section 2: Tham số tấn công */}
          <Card title="2. Tham số tấn công">
            <div className="space-y-4 text-xs">
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
                <div className="flex justify-between text-[10px] text-slate-400 mt-1">
                  <span>0.00 (Nhẹ)</span>
                  <span>0.50</span>
                  <span>1.00 (Mạnh nhất)</span>
                </div>
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
                <div className="flex justify-between text-[10px] text-slate-400 mt-1">
                  <span>1/255</span>
                  <span>8/255</span>
                  <span>16/255</span>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-2">
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
                <div>
                  <label className="text-slate-500 font-medium block mb-1">Random seed</label>
                  <input
                    type="number"
                    value={randomSeed}
                    onChange={(e) => setRandomSeed(parseInt(e.target.value) || 0)}
                    className="w-full px-2.5 py-1.5 rounded border border-slate-300 font-mono text-slate-800"
                  />
                </div>
              </div>

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

          {/* Section 3 & 4: Ràng buộc & Cài đặt Patch */}
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
                    <option value="full">Toàn bộ ảnh (Full image)</option>
                    <option value="roi">Vùng quan tâm (ROI bounding box)</option>
                  </select>
                </div>
                <div>
                  <label className="text-slate-500 font-medium block mb-1">Target confidence</label>
                  <input
                    type="text"
                    defaultValue="0.50"
                    className="w-full p-1.5 rounded border border-slate-300 font-mono"
                  />
                </div>
              </div>
            </Card>

            <Card title="4. Thiết lập Patch (Nếu có)">
              <div className="space-y-2 text-xs">
                <div>
                  <label className="text-slate-500 font-medium block mb-1">Kích thước Patch</label>
                  <select
                    value={patchSize}
                    onChange={(e) => setPatchSize(e.target.value)}
                    className="w-full p-1.5 rounded border border-slate-300 bg-white"
                  >
                    <option value="32 × 32">32 × 32 px (Nhỏ)</option>
                    <option value="64 × 64">64 × 64 px (Chuẩn)</option>
                    <option value="128 × 128">128 × 128 px (Lớn)</option>
                  </select>
                </div>
                <div>
                  <label className="text-slate-500 font-medium block mb-1">Vị trí dán</label>
                  <select
                    value={patchPos}
                    onChange={(e) => setPatchPos(e.target.value)}
                    className="w-full p-1.5 rounded border border-slate-300 bg-white"
                  >
                    <option value="bottom_right">Góc dưới phải</option>
                    <option value="center">Trung tâm vật thể</option>
                    <option value="random">Ngẫu nhiên</option>
                  </select>
                </div>
              </div>
            </Card>
          </div>

          {/* Section 5: Kiểm tra an toàn */}
          <Card title="5. Kiểm tra an toàn & xác thực">
            <div className="space-y-2 text-xs">
              <label className="flex items-center gap-2">
                <input type="checkbox" defaultChecked className="rounded text-blue-600" />
                <span className="text-slate-700">✓ Kiểm tra ràng buộc ε thực tế trên GPU</span>
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" defaultChecked className="rounded text-blue-600" />
                <span className="text-slate-700">✓ Kiểm tra NaN / Inf tensor gradient</span>
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" defaultChecked className="rounded text-blue-600" />
                <span className="text-slate-700">✓ Giới hạn thay đổi tối đa per-pixel: 0.15</span>
              </label>
            </div>
          </Card>
        </div>

        {/* RIGHT: Attack Summary, Queue & Live Preview (4 Columns) */}
        <div className="xl:col-span-4 space-y-4 sticky top-[74px]">
          {/* Attack Summary */}
          <Card title="Tóm tắt tấn công">
            <div className="space-y-1.5 text-xs divide-y divide-slate-100 font-medium mb-3">
              <div className="flex justify-between py-1">
                <span className="text-slate-500">Thuật toán</span>
                <span className="font-bold text-blue-600">{selectedAttackId.toUpperCase()}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-500">Chế độ mục tiêu</span>
                <span className="text-slate-800">{targetMode === "untargeted" ? "Không mục tiêu" : "Có mục tiêu"}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-500">Norm & Epsilon</span>
                <span className="font-mono text-red-600 font-bold">L∞ (ε = {epsilonNum}/255)</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-500">Step size & Iters</span>
                <span className="font-mono text-slate-800">{stepSize} ({iterations} iters)</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-500">Vùng áp dụng</span>
                <span className="text-slate-800">{scope === "full" ? "Toàn bộ ảnh" : "ROI"}</span>
              </div>
            </div>

            {/* Attack Queue */}
            <div className="mb-3">
              <div className="text-[11px] font-bold text-slate-500 mb-1.5">
                Thứ tự chuỗi tấn công (Queue)
              </div>
              <div className="space-y-1.5">
                {attackQueue.map((item, idx) => (
                  <div
                    key={item.id}
                    className="p-2 rounded bg-slate-50 border border-slate-200 flex items-center justify-between text-xs font-mono"
                  >
                    <div className="flex items-center gap-2">
                      <GripVertical className="w-3.5 h-3.5 text-slate-400" />
                      <span className="font-bold text-blue-600">{idx + 1}. {item.name}</span>
                      <span className="text-slate-500 text-[10px]">ε={item.eps}</span>
                    </div>
                    {attackQueue.length > 1 && (
                      <button
                        type="button"
                        onClick={() => handleRemoveQueue(item.id)}
                        className="text-red-500 hover:text-red-700"
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
                Xem trước tấn công (Live Preview)
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="h-24 rounded bg-slate-200 border border-slate-300 flex items-center justify-center text-slate-500 text-[10px] font-medium text-center p-1">
                  Ảnh gốc
                </div>
                <div className="h-24 rounded bg-slate-300 border border-red-300 flex items-center justify-center text-red-700 text-[10px] font-medium text-center p-1 relative overflow-hidden">
                  Ảnh sau PGD (Nhiễu)
                  <span className="absolute bottom-1 right-1 px-1 rounded bg-red-600 text-white text-[8px]">
                    +noise
                  </span>
                </div>
              </div>
              <div className="mt-1.5 text-[10px] text-emerald-600 font-medium">
                ✓ Dự kiến thay đổi: L∞ ≤ {(epsilonNum/255).toFixed(4)} ({epsilonNum}/255)
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
              {isExecuting ? "Đang chạy tấn công..." : "▶ Bắt đầu tấn công"}
            </Button>
          </Card>
        </div>
      </div>
    </div>
  );
}
