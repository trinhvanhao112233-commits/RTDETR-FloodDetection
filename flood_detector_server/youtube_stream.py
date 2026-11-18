"""Read public YouTube video frames in memory, without saving a video file."""

import base64
import io
import re
import shutil
import time
from urllib.parse import parse_qs, urlsplit

import cv2
from detection_result import detection_fields


def canonical_youtube_url(url):
    parts = urlsplit(str(url).strip())
    if parts.scheme not in ('http', 'https') or parts.username or parts.password or parts.port:
        raise ValueError('Hãy nhập link video YouTube hợp lệ.')
    host = (parts.hostname or '').lower()
    segments = parts.path.strip('/').split('/')
    if host == 'youtu.be':
        video_id = segments[0]
    elif host in ('youtube.com', 'www.youtube.com', 'm.youtube.com', 'music.youtube.com'):
        if parts.path == '/watch':
            video_id = parse_qs(parts.query).get('v', [''])[0]
        elif len(segments) == 2 and segments[0] in ('shorts', 'live', 'embed'):
            video_id = segments[1]
        else:
            video_id = ''
    else:
        video_id = ''
    if not re.fullmatch(r'[A-Za-z0-9_-]{11}', video_id):
        raise ValueError('Chỉ hỗ trợ link video YouTube, Shorts hoặc livestream cụ thể.')
    return f'https://www.youtube.com/watch?v={video_id}'


def resolve_youtube(url):
    from yt_dlp import YoutubeDL

    options = {
        'quiet': True, 'no_warnings': True, 'noplaylist': True,
        'skip_download': True, 'cachedir': False,
        'socket_timeout': 15, 'retries': 1, 'extractor_retries': 1,
        # HLS works for the supplied video without downloading/merging MP4 tracks.
        'format': 'bestvideo[protocol=m3u8_native][height<=720][vcodec^=avc1]'
                  '/best[protocol=m3u8_native][height<=720]'
                  '/bestvideo[protocol=m3u8_native][height<=720]',
    }
    if shutil.which('node'):
        options['js_runtimes'] = {'node': {}}
    with YoutubeDL(options) as downloader:
        info = downloader.extract_info(canonical_youtube_url(url), download=False)
    if not info or not info.get('url'):
        raise ValueError('Không tìm được luồng video công khai có thể đọc.')
    return info


class RemoteStream(io.RawIOBase):
    """Use Python TLS for HLS manifests/segments (works on Windows)."""

    def __init__(self, session, url, headers):
        parts = urlsplit(url)
        host = (parts.hostname or '').lower()
        if parts.scheme != 'https' or not (
            host.endswith('.googlevideo.com') or host.endswith('.youtube.com')
        ):
            raise ValueError('Nguồn phát YouTube không được hỗ trợ.')
        self.response = session.get(url, headers=headers, stream=True,
                                    timeout=(10, 15), allow_redirects=False)
        try:
            self.response.raise_for_status()
            if self.response.is_redirect:
                raise ValueError('Không đọc được nguồn phát chuyển hướng.')
        except Exception:
            self.response.close()
            raise
        self.response.raw.decode_content = True

    def readable(self):
        return True

    def read(self, size=-1):
        return self.response.raw.read(size)

    def close(self):
        if hasattr(self, 'response'):
            self.response.close()
        super().close()


def youtube_events(url, engine, fps, stop):
    import av
    import requests

    info = resolve_youtube(url)
    if stop.is_set():
        return
    yield {'status': 'loading', 'message': f'Đang mở: {info.get("title", "YouTube")}',
           'title': info.get('title', 'YouTube')}
    with requests.Session() as session:
        def open_remote(remote_url, flags, options):
            if flags & 2:
                raise ValueError('Luồng video chỉ cho phép đọc.')
            return RemoteStream(session, remote_url, info.get('http_headers', {}))

        with av.open(info['url'], format='hls', io_open=open_remote, timeout=(20, 20)) as container:
            next_media_time = 0.0
            first_media_time = None
            started = time.monotonic()
            frames = 0
            for decoded in container.decode(video=0):
                if stop.is_set():
                    return
                media_time = float(decoded.time or 0)
                if first_media_time is None:
                    first_media_time = media_time
                    next_media_time = media_time
                    started = time.monotonic()
                if media_time + 1e-6 < next_media_time:
                    continue
                if stop.wait(max(0, media_time - first_media_time - (time.monotonic() - started))):
                    return
                next_media_time = media_time + 1 / fps
                frame = decoded.to_ndarray(format='bgr24')
                candidates, latency = engine.predict(frame, include_candidates=True)
                fields = detection_fields(candidates)
                ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
                if not ok:
                    raise RuntimeError('Không thể hiển thị khung hình video.')
                frames += 1
                yield {
                    'status': 'success', 'timestamp': time.time(), 'latency_ms': latency,
                    'fps': round(frames / max(0.001, time.monotonic() - started), 1),
                    **fields,
                    'frame': base64.b64encode(encoded).decode('ascii'),
                    'video_time': max(0, media_time - first_media_time),
                    'metadata': {'frame_width': frame.shape[1], 'frame_height': frame.shape[0],
                                 'objects_count': len(fields['detections'])},
                }
            if not frames and not stop.is_set():
                raise RuntimeError('Nguồn phát không trả về khung hình video.')


def next_event(events):
    # StopIteration must not escape through an asyncio Future.
    return next(events, None)
