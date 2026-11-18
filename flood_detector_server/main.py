import base64
import asyncio
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional, Union

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
from youtube_stream import canonical_youtube_url, youtube_events, next_event
from detection_result import detection_fields
from pytorch_engine import RTDETRInferenceEngine

# ==============================================================================
# CẤU HÌNH HỆ THỐNG & HYPERPARAMETERS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "flood_detector_frontend")


engine: Optional[RTDETRInferenceEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("🚀 Initializing AI Inference Service...")
    try:
        engine = RTDETRInferenceEngine(model_path=MODEL_PATH)
        engine.load_model()
    except FileNotFoundError as fnf_err:
        print(f"⚠️ CẢNH BÁO: {fnf_err}")
        print("ℹ️ Server vẫn đang chạy. Đặt best.pt vào đúng đường dẫn rồi khởi động lại server.")
    yield
    print("🛑 Releasing resources and shutting down...")
    engine = None


# ==============================================================================
# FASTAPI APPLICATION & ROUTING
# ==============================================================================
app = FastAPI(
    title="Real-Time Flood Detection Server (RT-DETR)",
    version="1.0.0",
    lifespan=lifespan
)

# Kích hoạt CORS hỗ trợ client Vue.js chạy ở host/port khác
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode_image_data(message: Union[str, bytes]) -> Optional[np.ndarray]:
    """
    Giải mã khung hình dạng Binary hoặc Base64 thành NumPy Array (BGR) trực tiếp trong RAM.
    """
    try:
        if isinstance(message, bytes):
            # 1. Định dạng Binary (Buffer trực tiếp từ canvas / blob)
            img_buf = np.frombuffer(message, dtype=np.uint8)
            img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
            return img

        elif isinstance(message, str):
            # 2. Định dạng Base64 (DataURL ví dụ: 'data:image/jpeg;base64,...')
            if "," in message:
                message = message.split(",", 1)[1]
            img_bytes = base64.b64decode(message)
            img_buf = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
            return img
    except Exception as e:
        print(f"⚠️ Image decode error: {e}")
    return None


@app.websocket("/ws/flood-detection")
async def websocket_flood_detection(websocket: WebSocket):
    """
    Endpoint WebSocket đón nhận frame liên tục từ Vue.js client,
    suy luận real-time và trả kết quả Bounding Box, FPS, latency.
    """
    await websocket.accept()
    client_info = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"
    print(f"🔌 Client connected: {client_info}")

    try:
        while True:
            # Nhận message từ client (hỗ trợ binary hoặc text base64)
            data = await websocket.receive()
            if data['type'] == 'websocket.disconnect':
                break

            raw_frame_data = None
            if "bytes" in data and data["bytes"]:
                raw_frame_data = data["bytes"]
            elif "text" in data and data["text"]:
                raw_frame_data = data["text"]

            if raw_frame_data is None:
                continue

            # Kiểm tra trạng thái mô hình
            if engine is None or not engine.is_ready:
                await websocket.send_json({
                    "status": "error",
                    "message": "Mô hình chưa được tải. Hãy đặt best.pt trong thư mục server."
                })
                continue

            # 1. Giải mã frame sang ma trận ảnh OpenCV
            frame = decode_image_data(raw_frame_data)
            if frame is None:
                await websocket.send_json({
                    "status": "error",
                    "message": "Invalid or corrupted image frame payload."
                })
                continue

            # 2. Thực thi suy luận với RT-DETR
            candidates, latency_ms = await asyncio.to_thread(engine.predict, frame, include_candidates=True)
            fields = detection_fields(candidates)
            fps = round(1000.0 / latency_ms, 1) if latency_ms > 0 else 0.0

            # 3. Phản hồi JSON ngay lập tức cho client
            response_payload = {
                "status": "success",
                "timestamp": time.time(),
                "latency_ms": latency_ms,
                "fps": fps,
                **fields,
                "metadata": {
                    "frame_width": frame.shape[1],
                    "frame_height": frame.shape[0],
                    "objects_count": len(fields['detections'])
                }
            }
            await websocket.send_json(response_payload)

    except WebSocketDisconnect:
        print(f"🔌 Client disconnected: {client_info}")
    except Exception as exc:
        print(f"❌ WebSocket Session Exception: {exc}")
        try:
            await websocket.close()
        except Exception:
            pass




@app.websocket("/ws/youtube-detection")
async def websocket_youtube_detection(websocket: WebSocket):
    await websocket.accept()
    stop = threading.Event()
    events = None
    pending = receiver = None
    try:
        config = await asyncio.wait_for(websocket.receive_json(), timeout=15)
        url = canonical_youtube_url(config.get('url', ''))
        fps = float(config.get('fps', 5))
        if not np.isfinite(fps) or not 1 <= fps <= 10:
            raise ValueError('FPS YouTube phải nằm trong khoảng 1–10.')
        if engine is None or not engine.is_ready:
            raise ValueError('Mô hình chưa được tải. Kiểm tra file best.pt trên server.')
        await websocket.send_json({'status': 'loading', 'message': 'Đang kết nối YouTube…'})
        events = youtube_events(url, engine, fps, stop)
        while not stop.is_set():
            pending = asyncio.create_task(asyncio.to_thread(next_event, events))
            receiver = asyncio.create_task(websocket.receive_json())
            done, _ = await asyncio.wait((pending, receiver), return_when=asyncio.FIRST_COMPLETED)
            if receiver in done:
                receiver.result()  # Raise on disconnect; any message stops this stream.
                break
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
            payload = pending.result()
            if payload is None:
                await websocket.send_json({'status': 'ended', 'message': 'Đã phát hết video.'})
                break
            await websocket.send_json(payload)
            if payload['status'] == 'success':
                # One frame in flight: wait until the browser displays it.
                reply = await asyncio.wait_for(websocket.receive_json(), timeout=30)
                if reply.get('type') != 'next':
                    break
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        message = str(exc) if isinstance(exc, ValueError) else (
            'Không đọc được video YouTube. Video có thể bị giới hạn hoặc nguồn phát đã thay đổi. '
            'Thử link công khai khác hoặc cập nhật yt-dlp.'
        )
        print(f'YouTube stream error: {type(exc).__name__}')
        try:
            await websocket.send_json({'status': 'error', 'message': message})
        except Exception:
            pass
    finally:
        stop.set()
        if receiver:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
        # Let an outstanding network read finish before closing its generator.
        if pending:
            await asyncio.gather(pending, return_exceptions=True)
        if events:
            await asyncio.to_thread(events.close)
        try:
            await websocket.close()
        except Exception:
            pass


@app.post("/api/detect")
async def api_detect_image(file: Optional[UploadFile] = File(None), request: Request = None):
    """
    REST API suy luận ngập lụt từ file ảnh tải lên (Multipart Form hoặc Raw Body).
    """
    if engine is None or not engine.is_ready:
        raise HTTPException(status_code=503, detail="Mô hình chưa được tải. Hãy đặt best.pt vào thư mục server.")

    raw_bytes = None
    if file is not None:
        raw_bytes = await file.read()
    elif request is not None:
        raw_bytes = await request.body()

    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Không tìm thấy dữ liệu ảnh trong request.")

    frame = decode_image_data(raw_bytes)
    if frame is None:
        raise HTTPException(status_code=400, detail="Không thể giải mã định dạng ảnh.")

    candidates, latency_ms = await asyncio.to_thread(engine.predict, frame, include_candidates=True)
    fields = detection_fields(candidates)
    fps = round(1000.0 / latency_ms, 1) if latency_ms > 0 else 0.0

    return {
        "status": "success",
        "timestamp": time.time(),
        "latency_ms": latency_ms,
        "fps": fps,
        **fields,
        "metadata": {
            "frame_width": frame.shape[1],
            "frame_height": frame.shape[0],
            "objects_count": len(fields['detections'])
        }
    }


# ==============================================================================
# TỰ ĐỘNG PHỤC VỤ FRONTEND TĨNH (HTML/CSS/JS)
# ==============================================================================
if os.path.exists(FRONTEND_DIR):
    css_dir = os.path.join(FRONTEND_DIR, "css")
    js_dir = os.path.join(FRONTEND_DIR, "js")
    samples_dir = os.path.join(FRONTEND_DIR, "samples")

    if os.path.exists(css_dir):
        app.mount("/css", StaticFiles(directory=css_dir), name="css")
    if os.path.exists(js_dir):
        app.mount("/js", StaticFiles(directory=js_dir), name="js")
    if os.path.exists(samples_dir):
        app.mount("/samples", StaticFiles(directory=samples_dir), name="samples")

    @app.get("/")
    def serve_frontend_index():
        index_path = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "Frontend index.html not found"}


@app.get("/health")
def health_check():
    """Endpoint kiểm tra sức khỏe của dịch vụ suy luận."""
    is_ready = engine is not None and engine.is_ready
    return {
        "status": "online",
        "model_loaded": is_ready,
        "providers": ["PyTorchCUDA" if engine.device != "cpu" else "PyTorchCPU"] if is_ready else [],
        "runtime": "PyTorch / Ultralytics",
        "device": engine.device if is_ready else None,
        "gpu_name": engine.gpu_name if is_ready else None,
        "model_path": engine.model_path if engine else MODEL_PATH
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        access_log=False  # Tắt access log console để tránh nghẽn I/O khi frame rate cao
    )
