"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import {
  RotateCcw,
  Maximize2,
  Sliders,
  Eye,
  Grid,
  Activity,
  AlertTriangle,
} from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Project a 3D LiDAR point (in Velodyne frame: [x, y, z]) into 2D camera pixel coordinates (u, v).
 * Uses real KITTI calibration schema:
 * - Tr_velo_to_cam (4x4 or 3x4 transform)
 * - R_rect (3x3 or 4x4 rectification matrix)
 * - P2 (3x4 camera projection matrix)
 *
 * @param {Array<number>} point3d - [x, y, z] in LiDAR coordinates
 * @param {Object} calib - Calibration matrices { P2, R_rect, Tr_velo_to_cam }
 * @returns {{ u: number, v: number, depth: number } | null} - Projected pixel or null if behind camera / invalid
 */
export function projectKitti3dTo2d(point3d, calib) {
  if (!calib?.P2 || !calib?.Tr_velo_to_cam) {
    return null;
  }

  const [x, y, z] = point3d;
  const Tr = calib.Tr_velo_to_cam;
  const R_rect = calib.R_rect || [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
  const P2 = calib.P2;

  // 1. Transform LiDAR -> Camera coordinates: X_cam = Tr * [x, y, z, 1]^T
  const xc = Tr[0][0] * x + Tr[0][1] * y + Tr[0][2] * z + Tr[0][3];
  const yc = Tr[1][0] * x + Tr[1][1] * y + Tr[1][2] * z + Tr[1][3];
  const zc = Tr[2][0] * x + Tr[2][1] * y + Tr[2][2] * z + Tr[2][3];

  // 2. Rectify: X_rect = R_rect * X_cam
  const xr = R_rect[0][0] * xc + R_rect[0][1] * yc + R_rect[0][2] * zc;
  const yr = R_rect[1][0] * xc + R_rect[1][1] * yc + R_rect[1][2] * zc;
  const zr = R_rect[2][0] * xc + R_rect[2][1] * yc + R_rect[2][2] * zc;

  // Points behind camera lens have zr <= 0
  if (zr <= 0.1) {
    return null;
  }

  // 3. Project to pixels: [u*w, v*w, w]^T = P2 * [xr, yr, zr, 1]^T
  const u_w = P2[0][0] * xr + P2[0][1] * yr + P2[0][2] * zr + P2[0][3];
  const v_w = P2[1][0] * xr + P2[1][1] * yr + P2[1][2] * zr + P2[1][3];
  const w = P2[2][0] * xr + P2[2][1] * yr + P2[2][2] * zr + P2[2][3];

  if (w <= 0.001) {
    return null;
  }

  return {
    u: Math.round(u_w / w),
    v: Math.round(v_w / w),
    depth: zr,
  };
}

/**
 * Isometric camera projection helper for point cloud canvas rendering.
 */
function projectIso(x, y, z, cosYaw, sinYaw, cosPitch, sinPitch, scale, originX, originY) {
  const rx = cosYaw * x - sinYaw * y;
  const ry = sinYaw * x + cosYaw * y;
  const rz = z;

  const isoX = rx;
  const isoY = ry * sinPitch - rz * cosPitch;

  return {
    px: originX + isoX * scale,
    py: originY - isoY * scale,
  };
}

/**
 * Render a real 3D oriented bounding box with 8 vertices and 12 wireframe edges.
 */
function renderOrientedBox3D(ctx, box, cosYaw, sinYaw, cosPitch, sinPitch, scale, originX, originY) {
  const l = box.length || box.dx || 4.0;
  const w = box.width || box.dy || 2.0;
  const h = box.height || box.dz || 1.6;
  const boxYaw = box.yaw || 0.0;
  const cosB = Math.cos(boxYaw);
  const sinB = Math.sin(boxYaw);

  const halfL = l / 2;
  const halfW = w / 2;
  const halfH = h / 2;

  // Local 8 corners
  const localCorners = [
    [-halfL, -halfW, -halfH],
    [halfL, -halfW, -halfH],
    [halfL, halfW, -halfH],
    [-halfL, halfW, -halfH],
    [-halfL, -halfW, halfH],
    [halfL, -halfW, halfH],
    [halfL, halfW, halfH],
    [-halfL, halfW, halfH],
  ];

  // Rotate by box yaw and project
  const projectedCorners = localCorners.map(([dx, dy, dz]) => {
    const rx = dx * cosB - dy * sinB + box.x;
    const ry = dx * sinB + dy * cosB + box.y;
    const rz = dz + box.z;
    return projectIso(rx, ry, rz, cosYaw, sinYaw, cosPitch, sinPitch, scale, originX, originY);
  });

  const edges = [
    [0, 1], [1, 2], [2, 3], [3, 0], // Bottom
    [4, 5], [5, 6], [6, 7], [7, 4], // Top
    [0, 4], [1, 5], [2, 6], [3, 7], // Vertical pillars
  ];

  ctx.beginPath();
  edges.forEach(([i, j]) => {
    const p1 = projectedCorners[i];
    const p2 = projectedCorners[j];
    ctx.moveTo(p1.px, p1.py);
    ctx.lineTo(p2.px, p2.py);
  });
  ctx.stroke();

  // Front face cross to indicate heading
  ctx.beginPath();
  ctx.strokeStyle = "rgba(239, 68, 68, 0.8)";
  const f1 = projectedCorners[1];
  const f2 = projectedCorners[6];
  const f3 = projectedCorners[2];
  const f4 = projectedCorners[5];
  ctx.moveTo(f1.px, f1.py);
  ctx.lineTo(f2.px, f2.py);
  ctx.moveTo(f3.px, f3.py);
  ctx.lineTo(f4.px, f4.py);
  ctx.stroke();
  ctx.strokeStyle = "#10B981";
}

/**
 * Interactive 3D Point Cloud Canvas Viewer.
 */
export default function PointCloudViewer({
  points, // Float32Array or Array of [x, y, z, intensity]
  boxes3d = [],
  calibration = null,
  isLoading = false,
  error = null,
  maxPointsBudget = 50000,
  className = "",
}) {
  const canvasRef = useRef(null);
  const [pointBudget, setPointBudget] = useState(maxPointsBudget);
  const [fps, setFps] = useState(60);
  const [showGrid, setShowGrid] = useState(true);
  const [showBoxes, setShowBoxes] = useState(true);
  const [cameraZoom, setCameraZoom] = useState(1.0);
  const [cameraRotation, setCameraRotation] = useState({ pitch: 0.6, yaw: 0.0 });
  const isDraggingRef = useRef(false);
  const lastMouseRef = useRef({ x: 0, y: 0 });

  // Reset camera
  const handleResetCamera = useCallback(() => {
    setCameraZoom(1.0);
    setCameraRotation({ pitch: 0.6, yaw: 0.0 });
  }, []);

  // Mouse drag orbit controls
  const handleMouseDown = (e) => {
    isDraggingRef.current = true;
    lastMouseRef.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e) => {
    if (!isDraggingRef.current) return;
    const dx = e.clientX - lastMouseRef.current.x;
    const dy = e.clientY - lastMouseRef.current.y;
    lastMouseRef.current = { x: e.clientX, y: e.clientY };

    setCameraRotation((prev) => ({
      yaw: prev.yaw + dx * 0.008,
      pitch: Math.max(0.05, Math.min(Math.PI / 2 - 0.05, prev.pitch + dy * 0.008)),
    }));
  };

  const handleMouseUp = () => {
    isDraggingRef.current = false;
  };

  const handleWheel = (e) => {
    e.preventDefault();
    setCameraZoom((prev) => Math.max(0.2, Math.min(5.0, prev * (e.deltaY > 0 ? 0.9 : 1.1))));
  };

  // Render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let frameId;
    let lastTime = performance.now();
    let frameCount = 0;

    const render = (time) => {
      frameCount++;
      if (time - lastTime >= 1000) {
        setFps(Math.round((frameCount * 1000) / (time - lastTime)));
        frameCount = 0;
        lastTime = time;
      }

      const width = (canvas.width = canvas.parentElement?.clientWidth || 600);
      const height = (canvas.height = canvas.parentElement?.clientHeight || 450);

      // Background
      ctx.fillStyle = "#0B0F19";
      ctx.fillRect(0, 0, width, height);

      if (isLoading || error || !points || points.length === 0) {
        return;
      }

      const originX = width / 2;
      const originY = height * 0.65;
      const scale = Math.min(width, height) * 0.08 * cameraZoom;

      const cosYaw = Math.cos(cameraRotation.yaw);
      const sinYaw = Math.sin(cameraRotation.yaw);
      const cosPitch = Math.cos(cameraRotation.pitch);
      const sinPitch = Math.sin(cameraRotation.pitch);

      // 1. Draw Ground Grid
      if (showGrid) {
        ctx.strokeStyle = "rgba(75, 85, 99, 0.3)";
        ctx.lineWidth = 1;
        for (let r = -40; r <= 40; r += 10) {
          // X-parallel lines
          const p1 = projectIso(-40, r, 0, cosYaw, sinYaw, cosPitch, sinPitch, scale, originX, originY);
          const p2 = projectIso(40, r, 0, cosYaw, sinYaw, cosPitch, sinPitch, scale, originX, originY);
          ctx.beginPath();
          ctx.moveTo(p1.px, p1.py);
          ctx.lineTo(p2.px, p2.py);
          ctx.stroke();
        }
      }

      // 2. Point Downsampling LOD
      const totalPoints = Math.floor(points.length / (Array.isArray(points[0]) ? 1 : 4));
      const step = Math.max(1, Math.ceil(totalPoints / pointBudget));

      // 3. Draw Points
      ctx.fillStyle = "#38BDF8";
      for (let i = 0; i < totalPoints; i += step) {
        let x, y, z;
        if (Array.isArray(points[i])) {
          [x, y, z] = points[i];
        } else {
          x = points[i * 4];
          y = points[i * 4 + 1];
          z = points[i * 4 + 2];
        }

        if (!isFinite(x) || !isFinite(y) || !isFinite(z)) continue;

        const p = projectIso(x, y, z, cosYaw, sinYaw, cosPitch, sinPitch, scale, originX, originY);
        if (p.px >= 0 && p.px < width && p.py >= 0 && p.py < height) {
          ctx.fillRect(p.px, p.py, 1.5, 1.5);
        }
      }

      // 4. Draw 3D Bounding Boxes
      if (showBoxes && boxes3d?.length > 0) {
        ctx.strokeStyle = "#10B981";
        ctx.lineWidth = 1.5;
        boxes3d.forEach((box) => {
          renderOrientedBox3D(ctx, box, cosYaw, sinYaw, cosPitch, sinPitch, scale, originX, originY);
        });
      }

      frameId = requestAnimationFrame(render);
    };

    frameId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(frameId);
  }, [points, boxes3d, showGrid, showBoxes, cameraZoom, cameraRotation, pointBudget, isLoading, error]);

  return (
    <div
      className={cn(
        "relative w-full h-[420px] rounded-xl overflow-hidden bg-[#0B0F19] border border-slate-800 select-none",
        className
      )}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onWheel={handleWheel}
    >
      {/* Canvas */}
      <canvas ref={canvasRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

      {/* Loading Overlay */}
      {isLoading && (
        <div className="absolute inset-0 bg-[#0B0F19]/80 flex flex-col items-center justify-center text-slate-300 gap-2">
          <Activity className="w-6 h-6 animate-spin text-blue-400" />
          <span className="text-sm font-medium">Đang tải Point Cloud...</span>
        </div>
      )}

      {/* Error / No Data State */}
      {!isLoading && (error || !points || points.length === 0) && (
        <div className="absolute inset-0 bg-[#0B0F19]/80 flex flex-col items-center justify-center text-slate-400 gap-2">
          <AlertTriangle className="w-6 h-6 text-amber-400" />
          <span className="text-sm font-semibold">{error || "Chưa có dữ liệu Point Cloud (No data)"}</span>
        </div>
      )}

      {/* Control Overlay Bar */}
      <div className="absolute top-3 left-3 right-3 flex items-center justify-between pointer-events-none">
        {/* Left: Info / FPS */}
        <div className="flex items-center gap-2 bg-slate-900/80 backdrop-blur-xs px-3 py-1.5 rounded-lg border border-slate-700/50 text-xs font-mono text-slate-300 pointer-events-auto">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            {fps} FPS
          </span>
          <span className="text-slate-600">|</span>
          <span>Budget: {pointBudget.toLocaleString()} pts</span>
        </div>

        {/* Right: Camera / View controls */}
        <div className="flex items-center gap-1.5 bg-slate-900/80 backdrop-blur-xs p-1 rounded-lg border border-slate-700/50 pointer-events-auto">
          <button
            type="button"
            onClick={() => setShowGrid((v) => !v)}
            title="Bật/Tắt Grid"
            className={cn(
              "p-1.5 rounded-md text-xs transition-colors min-h-[36px] min-w-[36px] flex items-center justify-center",
              showGrid ? "bg-blue-600 text-white" : "text-slate-400 hover:text-white"
            )}
          >
            <Grid className="w-4 h-4" />
          </button>
          <button
            type="button"
            onClick={() => setShowBoxes((v) => !v)}
            title="Bật/Tắt 3D Bounding Boxes"
            className={cn(
              "p-1.5 rounded-md text-xs transition-colors min-h-[36px] min-w-[36px] flex items-center justify-center",
              showBoxes ? "bg-emerald-600 text-white" : "text-slate-400 hover:text-white"
            )}
          >
            <Eye className="w-4 h-4" />
          </button>
          <button
            type="button"
            onClick={handleResetCamera}
            title="Đặt lại Camera"
            className="p-1.5 rounded-md text-slate-400 hover:text-white text-xs min-h-[36px] min-w-[36px] flex items-center justify-center"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
