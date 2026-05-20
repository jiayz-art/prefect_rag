"""FastAPI应用入口。"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router, set_rag_service, RAGService
from src.core.config import config


def create_app() -> FastAPI:
    """创建FastAPI应用。"""
    app = FastAPI(
        title="多模态RAG知识库问答系统",
        description="面向个人的多模态RAG知识库问答系统API",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    @app.on_event("startup")
    async def startup():
        print("[API] 正在初始化RAG服务...")
        service = RAGService()
        set_rag_service(service)

        # 尝试加载已有索引
        if service.load_index():
            print("[API] 索引加载成功，服务就绪")
        else:
            print("[API] 未找到已有索引，请通过 POST /api/v1/index 构建索引")

        # 清理过期缓存
        cleaned = service.cache.clear_expired()
        if cleaned:
            print(f"[API] 清理了 {cleaned} 条过期缓存")

    @app.on_event("shutdown")
    async def shutdown():
        print("[API] 服务关闭")

    return app


app = create_app()


def main():
    import uvicorn
    host = config.get("api.host", "0.0.0.0")
    port = config.get("api.port", 8000)
    print(f"[API] 启动服务 http://{host}:{port}")
    print(f"[API] 文档地址 http://{host}:{port}/docs")
    uvicorn.run("src.api.main:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    main()
