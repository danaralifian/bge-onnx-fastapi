from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from datetime import datetime

async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "statusCode": exc.status_code,
            "timestamp": datetime.utcnow().isoformat(),
            "message": exc.detail,
            "path": request.url.path,
        },
    )

async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "statusCode": 500,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Internal server error",
            "path": request.url.path,
        },
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "statusCode": 422,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Validation Error",
            "details": exc.errors(),  # Tambahkan detail kalau mau debug
            "path": request.url.path,
        },
    )