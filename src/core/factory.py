"""组件工厂 — 可插拔组件注册与获取，支持灵活替换模型/策略。"""
from typing import Any, Callable, Dict


class ComponentRegistry:
    """组件注册表，按类别管理组件。"""

    def __init__(self):
        self._registry: Dict[str, Dict[str, Callable]] = {}

    def register(self, category: str, name: str, factory: Callable[..., Any]):
        """注册组件工厂函数。"""
        if category not in self._registry:
            self._registry[category] = {}
        self._registry[category][name] = factory

    def get(self, category: str, name: str, **kwargs) -> Any:
        """获取组件实例。"""
        if category not in self._registry:
            raise KeyError(f"Unknown category: {category}")
        if name not in self._registry[category]:
            available = list(self._registry[category].keys())
            raise KeyError(f"Unknown {category} '{name}'. Available: {available}")
        return self._registry[category][name](**kwargs)

    def list(self, category: str = None) -> Dict[str, list]:
        """列出已注册的组件。"""
        if category:
            return {category: list(self._registry.get(category, {}).keys())}
        return {cat: list(components.keys()) for cat, components in self._registry.items()}


# 全局注册表实例
registry = ComponentRegistry()


# ===== 装饰器快捷注册 =====

def register_parser(name: str):
    """注册文档解析器。"""
    def decorator(fn):
        registry.register("parser", name, fn)
        return fn
    return decorator


def register_embedder(name: str):
    """注册Embedding模型。"""
    def decorator(fn):
        registry.register("embedder", name, fn)
        return fn
    return decorator


def register_llm(name: str):
    """注册LLM。"""
    def decorator(fn):
        registry.register("llm", name, fn)
        return fn
    return decorator


def register_reranker(name: str):
    """注册Reranker。"""
    def decorator(fn):
        registry.register("reranker", name, fn)
        return fn
    return decorator


def register_chunker(name: str):
    """注册Chunk策略。"""
    def decorator(fn):
        registry.register("chunker", name, fn)
        return fn
    return decorator
