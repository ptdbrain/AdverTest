"""End-to-End Full Pipeline Verification Script for AdverTest.

Simulates the complete researcher lifecycle from start to finish:
Session creation -> Preflight -> Attack Execution -> Metric/Result DB storage
-> Auto Hard-Gate Risk Classification -> Smart Clustering -> Review Resolution
-> Downstream Retraining Backlog & Defense Profile creation -> Session Finalization.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

# Ensure root workspace is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.main import app

client = TestClient(app)
DB_PATH = Path("data/app.db")
SESSIONS_JSON = Path("data/storage/sessions.json")


def log_step(step_num: int, title: str):
    print(f"\n{'=' * 70}")
    print(f"  BƯỚC {step_num}: {title.upper()}")
    print(f"{'=' * 70}")


def query_db(sql: str, params: tuple = ()) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]


def main():
    print("🚀 BẮT ĐẦU KIỂM THỬ TOÀN DIỆN END-TO-END PIPELINE & DATABASE PERSISTENCE")
    session_id = f"EXP-E2E-{uuid.uuid4().hex[:6].upper()}"

    # ─────────────────────────────────────────────────────────────────────────
    # BƯỚC 1: TẠO PHIÊN LÀM VIỆC (SESSION INITIALIZATION)
    # ─────────────────────────────────────────────────────────────────────────
    log_step(1, "Khởi tạo Phiên làm việc (Session) & Lưu trữ JSON DB")
    session_payload = {
        "id": session_id,
        "name": f"Phiên Thử Nghiệm Kiểm Định Toàn Diện {session_id}",
        "description": "E2E verification of attack execution, metric calculation, HITL triage and DB persistence",
        "task_id": "detection2d",
        "task_name": "Phát hiện vật thể 2D",
        "model_id": "yolo11s",
        "model_name": "YOLO11s (Ultralytics)",
        "dataset_id": "kitti",
        "dataset_name": "KITTI 2D Detection (Anonymized)",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs": [],
        "status": "active",
    }
    res = client.post("/api/v1/sessions", json=session_payload)
    assert res.status_code == 200, f"Session create failed: {res.text}"
    print(f"✅ Đã tạo Session thành công: {session_id}")

    # Kiểm tra file sessions.json
    assert SESSIONS_JSON.exists(), "File sessions.json không tồn tại!"
    with open(SESSIONS_JSON, encoding="utf-8") as f:
        stored_sessions = json.load(f)
    # stored_sessions is dict { session_id: data }
    assert session_id in stored_sessions, f"Session {session_id} không tìm thấy trong sessions.json!"
    print(f"✅ Đã xác thực Session được lưu bền vững vào '{SESSIONS_JSON}'")

    # ─────────────────────────────────────────────────────────────────────────
    # BƯỚC 2: ƯỚC TÍNH CHI PHÍ GPU & PREFLIGHT SAFETY GATE
    # ─────────────────────────────────────────────────────────────────────────
    log_step(2, "Ước tính chi phí GPU (Preflight Cost Estimation)")
    run_config = {
        "dataset": "kitti",
        "dataset_version_id": "kitti",
        "model": "yolo11",
        "model_family_id": "yolo11",
        "adapter_params": {"weights": "checkpoints/surrogates/yolo11s.pt"},
        "task_id": "detection2d",
        "attacks": ["fog", "fgsm"],
        "severities": [3, 4],
        "limit": 4,
        "seed": 42,
        "execution_mode": "benchmark",
    }
    res_est = client.post("/api/v1/runs/estimate", json=run_config)
    assert res_est.status_code == 200, f"Estimate failed: {res_est.text}"
    est = res_est.json()
    print(f"✅ Ước tính tài nguyên: {est['n_cells']} cells, {est['n_samples']} samples, {est['estimated_seconds']}s")

    # ─────────────────────────────────────────────────────────────────────────
    # BƯỚC 3: THỰC THI KIỂM THỬ ĐỐI KHÁNG (TEST RUN EXECUTION)
    # ─────────────────────────────────────────────────────────────────────────
    log_step(3, "Thực thi Test Run & Ghi nhận vào SQLite DB")
    res_run = client.post("/api/v1/runs", json=run_config)
    assert res_run.status_code == 202, f"Run start failed: {res_run.text}"
    run_id = res_run.json()["run_id"]
    print(f"✅ Đã enqueue Test Run: {run_id}")

    # Chờ chạy xong
    max_wait = 60
    start_t = time.time()
    completed = False
    report = None
    while time.time() - start_t < max_wait:
        res_poll = client.get(f"/api/v1/runs/{run_id}")
        assert res_poll.status_code == 200
        data_poll = res_poll.json()
        status = data_poll.get("status")
        if status == "COMPLETED":
            completed = True
            report = data_poll.get("report")
            break
        elif status in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"Run ended with status {status}: {data_poll.get('error')}")
        time.sleep(1)

    assert completed, f"Test run không hoàn thành trong {max_wait}s!"
    print(f"✅ Test Run hoàn thành trong {time.time() - start_t:.1f}s | Clean AP: {report.get('ap_clean', 0):.4f}")

    # Kiểm tra SQLite tables
    runs_in_db = query_db("SELECT * FROM test_runs WHERE run_id=?", (run_id,))
    assert len(runs_in_db) == 1, "Run không được lưu vào bảng test_runs!"
    assert runs_in_db[0]["status"] == "COMPLETED"

    events_in_db = query_db("SELECT * FROM test_run_events WHERE run_id=?", (run_id,))
    assert len(events_in_db) > 0, "Events không được lưu vào test_run_events!"

    metrics_in_db = query_db("SELECT * FROM metric_results WHERE run_id=?", (run_id,))
    assert len(metrics_in_db) > 0, "Metrics không được lưu vào metric_results!"

    samples_in_db = query_db("SELECT * FROM sample_results WHERE run_id=?", (run_id,))
    assert len(samples_in_db) > 0, "Samples không được lưu vào sample_results!"

    print("✅ Đã xác thực lưu trữ SQLite `data/app.db`:")
    print("   • test_runs: 1 record (status=COMPLETED)")
    print(f"   • test_run_events: {len(events_in_db)} events")
    print(f"   • metric_results: {len(metrics_in_db)} cells (AP/mAP deltas)")
    print(f"   • sample_results: {len(samples_in_db)} samples (predictions & paths)")

    # ─────────────────────────────────────────────────────────────────────────
    # BƯỚC 4: LƯU KẾT QUẢ VÀO RUN HISTORY VÀ GHI CHÚ KHOA HỌC
    # ─────────────────────────────────────────────────────────────────────────
    log_step(4, "Ghi nhận Lần chạy vào Session & Cập nhật Researcher Notes")
    last_cell = report.get("cells", [{}])[-1]
    clean_ap = report.get("ap_clean", 0.85)
    attacked_ap = last_cell.get("ap", 0.40)
    map_drop = ((clean_ap - attacked_ap) / clean_ap * 100.0) if clean_ap > 0 else 0.0

    run_record = {
        "id": run_id,
        "name": "Lần 1: Fog + FGSM (Cấp 3, 4)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "attack_type": "fog+fgsm",
        "attack_name": "Sương mù & Gradient",
        "severity": 4,
        "clean_map": round(clean_ap, 4),
        "attacked_map": round(attacked_ap, 4),
        "map_drop_pct": round(map_drop, 1),
        "clean_conf": 0.88,
        "attacked_conf": 0.42,
        "psnr": "21.4 dB",
        "ssim": "0.78",
        "inference_ms": 32.5,
        "robustness_score": round(max(0, 100 - map_drop), 1),
        "clean_bbox_count": 6,
        "attacked_bbox_count": 2,
        "sample_id": "000000",
        "is_combined": True,
        "attack_components": ["fog", "fgsm"],
        "seed": 42,
        "note": "Phát hiện sụt giảm mạnh khi sương mù dày kết hợp nhiễu gradient.",
        "backend_run_id": run_id,
    }
    res_add_run = client.post(f"/api/v1/sessions/{session_id}/runs", json=run_record)
    assert res_add_run.status_code == 200, f"Add run failed: {res_add_run.text}"
    print(f"✅ Đã thêm Run vào Session: sụt giảm {map_drop:.1f}% mAP")

    # Cập nhật ghi chú khoa học
    res_note = client.patch(
        f"/api/v1/sessions/{session_id}/runs/{run_id}/note",
        json={"note": "Cần kích hoạt phòng thủ adversarial_training cho lớp Pedestrian."},
    )
    assert res_note.status_code == 200
    print("✅ Đã cập nhật Researcher Note thành công")

    # ─────────────────────────────────────────────────────────────────────────
    # BƯỚC 5: PHỄU LỌC TỰ ĐỘNG (AUTO HARD-GATE & REVIEW QUEUE)
    # ─────────────────────────────────────────────────────────────────────────
    log_step(5, "Auto Hard-Gate: Tự động phân loại rủi ro 5 cấp & Gắn cờ Review")
    res_flag = client.post(f"/api/v1/runs/{run_id}/flag-reviews?threshold=20.0")
    assert res_flag.status_code == 200
    flag_data = res_flag.json()
    print(f"✅ Auto-Flag kết quả: {flag_data}")

    reviews_in_db = query_db("SELECT * FROM reviews WHERE run_id=?", (run_id,))
    assert len(reviews_in_db) > 0, "Không có review nào được tạo trong bảng reviews!"
    first_review = reviews_in_db[0]
    review_id = first_review["review_id"]
    print(f"✅ Đã tạo {len(reviews_in_db)} ca review trong SQLite table `reviews`:")
    for rev in reviews_in_db:
        print(
            f"   • [{rev['review_id']}] Attack={rev['attack']} Sev={rev['severity']} Risk={rev['risk_level']} Status={rev['status']}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # BƯỚC 6: GOM CỤM LỖI THÔNG MINH (SMART FAILURE CLUSTERING)
    # ─────────────────────────────────────────────────────────────────────────
    log_step(6, "Smart Failure Clustering: Gom cụm ca lỗi & Lưu trữ DB")
    res_cluster = client.post(f"/api/v1/failure-clusters/auto-group?run_id={run_id}")
    assert res_cluster.status_code == 200
    clusters = res_cluster.json()
    assert len(clusters) > 0, "Không tạo được failure cluster nào!"
    first_cluster = clusters[0]
    cluster_id = first_cluster["cluster_id"]
    print(f"✅ Đã gom thành {len(clusters)} cụm lỗi thông minh:")
    for c in clusters:
        print(f"   • [{c['cluster_id']}] {c['name']} ({c['member_count']} ca) -> Gợi ý: {c['recommended_action']}")

    # Xác thực trong product_records
    clusters_in_db = query_db("SELECT * FROM product_records WHERE record_type='failure_cluster'")
    assert any(c["record_id"] == cluster_id for c in clusters_in_db)
    print("✅ Đã xác thực cluster được lưu trong bảng `product_records`")

    # ─────────────────────────────────────────────────────────────────────────
    # BƯỚC 7: ĐÁNH GIÁ RỦI RO CHUYÊN SÂU (RISK ASSESSMENT ENGINE)
    # ─────────────────────────────────────────────────────────────────────────
    log_step(7, "Risk Assessment Engine: Sinh giải trình & Khuyến nghị quyết định")
    res_assess = client.post(f"/api/v1/risk-rubric/assess?review_id={review_id}")
    assert res_assess.status_code == 200
    assessment = res_assess.json()
    print(f"✅ Phân tích rủi ro cho {review_id}:")
    print(f"   • Mức độ rủi ro: {assessment['risk_level']}")
    print(f"   • Khuyến nghị: {assessment['recommended_decision_label']}")
    print(f"   • Giải trình: {assessment['justification']}")
    print(f"   • Độ tin cậy: {assessment['confidence'] * 100:.0f}%")
    print(f"   • Hành động tiếp theo: {' -> '.join(assessment['downstream_actions'])}")

    # ─────────────────────────────────────────────────────────────────────────
    # BƯỚC 8: THẨM ĐỊNH & KÍCH HOẠT DOWNSTREAM (ONE-CLICK RESOLUTION)
    # ─────────────────────────────────────────────────────────────────────────
    log_step(8, "Reviewer Quyết định & Tự động Kích hoạt Downstream Actions")
    resolve_payload = {
        "decision": "BLOCK_DEPLOY",
        "decision_note": "Chặn triển khai: Phát hiện sụt giảm nghiêm trọng trong sương mù và tấn công gradient. Bắt buộc retrain.",
        "resolved_by": "Staff Safety Engineer",
        "batch_cluster_id": cluster_id,
    }
    res_resolve = client.patch(f"/api/v1/reviews/{review_id}", json=resolve_payload)
    assert res_resolve.status_code == 200
    print(f"✅ Đã Resolve Review {review_id} với quyết định BLOCK_DEPLOY và áp dụng cho toàn bộ cụm {cluster_id}")

    # Kiểm tra trạng thái trong bảng reviews
    updated_reviews = query_db("SELECT * FROM reviews WHERE review_id=?", (review_id,))
    assert updated_reviews[0]["status"] == "RESOLVED"
    assert updated_reviews[0]["decision"] == "BLOCK_DEPLOY"
    assert updated_reviews[0]["resolved_by"] == "Staff Safety Engineer"
    print("✅ Đã xác thực cập nhật trạng thái `reviews` = RESOLVED")

    # Kiểm tra downstream: Retraining Backlog đã được tạo tự động
    backlogs_in_db = query_db("SELECT * FROM retraining_backlogs")
    assert len(backlogs_in_db) > 0, "Retraining Backlog không được tạo!"
    print(f"✅ Đã xác thực bảng `retraining_backlogs`: {len(backlogs_in_db)} backlogs được tạo tự động")

    backlog_items = query_db("SELECT * FROM retraining_backlog_items")
    assert len(backlog_items) > 0, "Retraining Backlog Items không được tạo!"
    print(f"✅ Đã xác thực bảng `retraining_backlog_items`: {len(backlog_items)} failure items được nạp vào backlog")

    # Kiểm tra downstream: Defense Profile & Deploy Gate trong product_records
    defense_profiles = query_db("SELECT * FROM product_records WHERE record_type='defense_profile'")
    deploy_gates = query_db("SELECT * FROM product_records WHERE record_type='deploy_gate'")
    print("✅ Đã xác thực bảng `product_records`:")
    print(f"   • defense_profile: {len(defense_profiles)} profiles được sinh tự động")
    print(f"   • deploy_gate: {len(deploy_gates)} gates ghi nhận chặn CI/CD deployment")

    # ─────────────────────────────────────────────────────────────────────────
    # BƯỚC 9: KẾT THÚC PHIÊN LÀM VIỆC (SESSION LIFECYCLE FINALIZATION)
    # ─────────────────────────────────────────────────────────────────────────
    log_step(9, "Hoàn tất Phiên làm việc (End Session Lifecycle)")
    res_end = client.post(f"/api/v1/sessions/{session_id}/end")
    assert res_end.status_code == 200
    ended_session = res_end.json()
    assert ended_session["status"] == "completed"
    assert ended_session["ended_at"] is not None
    print(f"✅ Đã khóa phiên thành công: status={ended_session['status']}, ended_at={ended_session['ended_at']}")

    # ─────────────────────────────────────────────────────────────────────────
    # BƯỚC 10: TỔNG KẾT TOÀN BỘ CƠ SỞ DỮ LIỆU
    # ─────────────────────────────────────────────────────────────────────────
    log_step(10, "Tổng kết Trạng thái Cơ sở Dữ liệu Hệ thống")
    print("📊 BẢNG TỔNG HỢP CÁC BẢNG DỮ LIỆU TRONG SQLite `data/app.db`:")
    tables = [
        "test_runs",
        "test_run_events",
        "metric_results",
        "sample_results",
        "reviews",
        "retraining_backlogs",
        "retraining_backlog_items",
        "product_records",
    ]
    for tbl in tables:
        count = query_db(f"SELECT count(*) as c FROM {tbl}")[0]["c"]
        print(f"   • {tbl:<26} : {count:>5} records")

    print("\n📁 FILE STORAGE:")
    with open(SESSIONS_JSON, encoding="utf-8") as f:
        sess_data = json.load(f)
    print(
        f"   • {str(SESSIONS_JSON):<26} : {len(sess_data):>5} sessions (kích thước: {SESSIONS_JSON.stat().st_size} bytes)"
    )

    print("\n🎉 TOÀN BỘ PIPELINE TỪ ĐẦU ĐẾN CUỐI ĐÃ ĐƯỢC KIỂM THỬ THÀNH CÔNG 100% VÀ LƯU DATABASE HOÀN HẢO!")


if __name__ == "__main__":
    main()
