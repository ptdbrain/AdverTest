"use client";

import { useEffect, useRef, useState } from "react";
import { useLanguage } from "@/context/LanguageContext";

/**
 * Bird's Eye View (BEV) Viewer for 3D bounding boxes and LiDAR.
 * 
 * Draws the ego vehicle at the origin (center bottom or center),
 * scales LiDAR points (if any) and 3D bounding boxes.
 */
export default function BevViewer({
  groundTruth,
  prediction,
  lidarPoints,
  range = 50, // default 50 meters
}) {
  const canvasRef = useRef(null);
  const { t } = useLanguage();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    
    // Handle resizing
    const resizeCanvas = () => {
      const parent = canvas.parentElement;
      if (parent) {
        canvas.width = parent.clientWidth;
        canvas.height = parent.clientHeight;
        draw();
      }
    };

    const draw = () => {
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);

      // We map the world coordinates to pixels.
      // Assuming X is right/left, Y is forward/backward
      // Typical KITTI LiDAR: X forward, Y left. Let's assume ego is at (0,0).
      // Let's set origin at center bottom.
      const originX = width / 2;
      const originY = height * 0.9;
      const pixelsPerMeter = Math.min(width, height) / (range * 2);

      const worldToPixel = (x, y) => {
        // Map world (x,y) to pixel (px, py)
        // Here we map y (forward) to -py, x (left) to -px
        return {
          px: originX - y * pixelsPerMeter,
          py: originY - x * pixelsPerMeter
        };
      };

      // Draw ego vehicle
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(originX, originY, 4, 0, 2 * Math.PI);
      ctx.fill();

      // Draw grid
      ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
      ctx.lineWidth = 1;
      for (let r = 10; r <= range; r += 10) {
        ctx.beginPath();
        ctx.arc(originX, originY, r * pixelsPerMeter, Math.PI, 2 * Math.PI);
        ctx.stroke();
      }

      // Draw ground truth
      if (groundTruth?.objects3d) {
        groundTruth.objects3d.forEach(box => {
          drawBox(ctx, box, "rgba(34, 197, 94, 0.8)", "rgba(34, 197, 94, 0.1)", worldToPixel, pixelsPerMeter);
        });
      }

      // Draw predictions
      if (prediction?.boxes3d) {
        prediction.boxes3d.forEach(box => {
          drawBox(ctx, box, "rgba(59, 130, 246, 0.8)", "rgba(59, 130, 246, 0.1)", worldToPixel, pixelsPerMeter);
        });
      }
    };

    const drawBox = (ctx, box, strokeColor, fillColor, worldToPixel, ppm) => {
      ctx.save();
      const center = worldToPixel(box.x, box.y);
      ctx.translate(center.px, center.py);
      // Rotation in 2D canvas is clockwise, yaw is usually counter-clockwise from X axis
      ctx.rotate(-box.yaw);
      
      const boxW = box.width * ppm;
      const boxL = box.length * ppm;
      
      ctx.beginPath();
      ctx.rect(-boxW / 2, -boxL / 2, boxW, boxL);
      ctx.fillStyle = fillColor;
      ctx.fill();
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw direction arrow
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(0, -boxL / 2);
      ctx.strokeStyle = "rgba(255,255,255,0.8)";
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.restore();
    };

    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();

    return () => window.removeEventListener("resize", resizeCanvas);
  }, [groundTruth, prediction, range]);

  return (
    <div style={{ width: "100%", height: "100%", position: "relative", background: "#0B0F17" }}>
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: "100%", display: "block" }}
      />
      <div style={{
        position: "absolute", bottom: 8, left: 8, 
        display: "flex", gap: 12, fontSize: 12, 
        fontFamily: "var(--font-mono)", color: "rgba(255,255,255,0.7)"
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <div style={{ width: 12, height: 12, background: "rgba(34, 197, 94, 0.8)" }}></div> GT
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <div style={{ width: 12, height: 12, background: "rgba(59, 130, 246, 0.8)" }}></div> Pred
        </div>
      </div>
    </div>
  );
}
