from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import json

class ResponseFormatterMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        try:
            response = await call_next(request)

            # Only format JSON responses
            if "application/json" in response.headers.get("content-type", ""):
                original_body = b""
                async for chunk in response.body_iterator:
                    original_body += chunk

                try:
                    data = json.loads(original_body)
                except json.JSONDecodeError:
                    data = original_body.decode()

                formatted = {
                    "status": response.status_code,
                    "data": data
                }

                return JSONResponse(content=formatted, status_code=response.status_code)

            return response
        except Exception as e:
            return JSONResponse(status_code=500, content={
                "status": 500,
                "error": str(e)
            })
