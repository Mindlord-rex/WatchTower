import os
import re
from typing import Generator
from fastapi import Request, HTTPException, status
from fastapi.responses import StreamingResponse

RANGE_HEADER_REGEX = re.compile(r"bytes=(\d+)-(\d*)")


def stream_video_file(path: str, request: Request) -> StreamingResponse:
    """
    Streams a video file supporting HTTP 206 Partial Content range requests.
    Enables native HTML5 video seeking and timestamp fragments (#t=...).
    """
    if not os.path.isfile(path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video file not found")

    file_size = os.path.getsize(path)
    range_header = request.headers.get("range")

    content_type = "video/mp4"
    if path.lower().endswith(".webm"):
        content_type = "video/webm"
    elif path.lower().endswith(".mov"):
        content_type = "video/quicktime"
    elif path.lower().endswith(".mkv"):
        content_type = "video/x-matroska"

    if range_header:
        match = RANGE_HEADER_REGEX.match(range_header)
        if match:
            start_str, end_str = match.groups()
            start = int(start_str)
            end = int(end_str) if end_str else file_size - 1

            if start >= file_size:
                raise HTTPException(
                    status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
                    detail="Requested Range Not Satisfiable"
                )

            end = min(end, file_size - 1)
            content_length = end - start + 1

            def range_generator() -> Generator[bytes, None, None]:
                with open(path, "rb") as f:
                    f.seek(start)
                    bytes_remaining = content_length
                    chunk_size = 1024 * 512  # 512 KB
                    while bytes_remaining > 0:
                        read_bytes = min(chunk_size, bytes_remaining)
                        data = f.read(read_bytes)
                        if not data:
                            break
                        bytes_remaining -= len(data)
                        yield data

            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
                "Content-Type": content_type,
            }
            return StreamingResponse(
                range_generator(),
                status_code=status.HTTP_206_PARTIAL_CONTENT,
                headers=headers
            )

    # Standard full file response
    def full_generator() -> Generator[bytes, None, None]:
        with open(path, "rb") as f:
            chunk_size = 1024 * 512
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                yield data

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
        "Content-Type": content_type,
    }
    return StreamingResponse(full_generator(), status_code=status.HTTP_200_OK, headers=headers)
