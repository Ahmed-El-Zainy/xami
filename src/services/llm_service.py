# services/llm_service.py
import asyncio
import time
from typing import List, Dict, Any, Optional
import json

import httpx
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.config_settings import settings
from src.config.logging_config import get_logger, log_execution_time, CustomLoggerTracker
from src.models.schemas import SearchResult, RAGResponse
from src.utilities.arabic_utils import arabic_processor
from src.services.resource_manager import resource_manager


try:
    custom = CustomLoggerTracker()
    _logger = custom.get_logger("ollama_llm_service")
    _logger.info("Custom Logger Start Working.....")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    _logger = logging.getLogger("ollama_llm_service")
    _logger.info("Using standard logger - custom logger not available")


class OllamaService:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.base_url = getattr(settings, 'OLLAMA_HOST', 'http://localhost:11434')
        self.model = getattr(settings, 'OLLAMA_MODEL', 'custom-qwen3_30b_a3b_Q3_K_S:latest')
        self._initialized = False

    def _build_ollama_options(self) -> Dict[str, Any]:
        """Conservative Ollama options to reduce memory footprint."""
        opts: Dict[str, Any] = {
            "num_ctx": int(getattr(settings, 'OLLAMA_NUM_CTX', 256)),
            "num_predict": int(getattr(settings, 'OLLAMA_NUM_PREDICT', 256)),
            "temperature": 0.2,
            "repeat_penalty": 1.1,
            "use_mmap": True,
            "use_mlock": False,
        }
        # Optional GPU offload control if provided in settings
        gpu_layers = getattr(settings, 'OLLAMA_GPU_LAYERS', None)
        if gpu_layers is not None:
            try:
                opts["gpu_layers"] = int(gpu_layers)
            except Exception:
                pass
        num_gpu = getattr(settings, 'OLLAMA_NUM_GPU', None)
        if num_gpu is not None:
            try:
                opts["num_gpu"] = int(num_gpu)
            except Exception:
                pass
        # Some backends honor low_vram
        if getattr(settings, 'OLLAMA_LOW_VRAM', None):
            opts["low_vram"] = True
        return opts

    async def initialize(self):
        if self._initialized:
            return
        try:
            await resource_manager.claim("llm")
            self.logger.info(f"Initializing Ollama LLM at {self.base_url} with model {self.model}")
            async with httpx.AsyncClient(timeout=120.0) as client: # Increased timeout to 120 seconds
                # Try a trivial generation to validate
                payload = {
                    "model": self.model,
                    "prompt": "hi",
                    "stream": False,
                    "options": self._build_ollama_options(),
                    "keep_alive": 0,
                }
                resp = await client.post(f"{self.base_url}/api/generate", json=payload)
                if resp.status_code != 200:
                    raise RuntimeError(f"Ollama generate failed: {resp.status_code} {resp.text}")
                _ = resp.json().get("response", "")
            self._initialized = True
            self.logger.info("Ollama LLM initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama LLM: {str(e)}")
            raise

    async def _generate_text(self, prompt: str, system: Optional[str] = None) -> str:
        try:
            await self.initialize()
            data = {
                "model": self.model,
                "prompt": (f"{system}\n\n{prompt}" if system else prompt),
                "stream": False,
                "options": self._build_ollama_options(),
                "keep_alive": 0,
            }
            async with httpx.AsyncClient(timeout=None) as client:
                resp = await client.post(f"{self.base_url}/api/generate", json=data)
                if resp.status_code != 200:
                    return f"عذراً، فشل التوليد: {resp.status_code}"
                out = resp.json()
                return out.get("response", "")
        except Exception as e:
            self.logger.error(f"Ollama generation error: {str(e)}")
            return f"عذراً، حدث خطأ في توليد الإجابة: {str(e)}"

    async def _build_rag_prompt(
        self,
        query: str,
        search_results: List[SearchResult],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        # Detect language
        query_language = arabic_processor.detect_language(query)

        # Build context
        context_parts = []
        for i, result in enumerate(search_results[:10], 1):
            context_parts.append(f"[مصدر {i}]\n{result.text}\n")
        context = "\n".join(context_parts)

        # History
        history_text = ""
        if conversation_history:
            parts = []
            for turn in conversation_history[-5:]:
                parts.append(f"المستخدم: {turn.get('user', '')}")
                parts.append(f"المساعد: {turn.get('assistant', '')}")
            history_text = "\n".join(parts)

        if query_language in ("ar", "mixed"):
            prompt = f"""
أنت مساعد ذكي متخصص في التعليم الثانوي العربي. أجب عن أسئلة الطلاب اعتماداً فقط على المصادر المرجعية التالية.

{'## سياق المحادثة السابقة:' + chr(10) + history_text + chr(10) if history_text else ''}

## المحتوى المرجعي:
{context}

## سؤال الطالب:
{query}

## تعليمات الإجابة:
1. اكتب الإجابة بالعربية الفصحى بدقة ووضوح
2. استخدم فقط المعلومات الموجودة في المحتوى المرجعي أعلاه
3. إذا لم تتوفر الإجابة في المصادر، اذكر ذلك صراحة
4. قدم أمثلة أو خطوات حل عند الإمكان
5. حافظ على تنظيم الإجابة باستخدام نقاط مرتبة

## الإجابة:
"""
        else:
            prompt = f"""
You are an assistant specialized in Arabic secondary education. Answer the student's question using ONLY the reference content below. Respond in Arabic.

{'## Previous Conversation Context:' + chr(10) + history_text + chr(10) if history_text else ''}

## Reference Content:
{context}

## Student Question:
{query}

## Answer Instructions:
1. Respond in Modern Standard Arabic clearly and accurately
2. Use only the information present in the reference content
3. If not found, explicitly say so
4. Provide examples or steps when possible
5. Use structured bullet points

## Answer:
"""
        return prompt

    async def _calculate_confidence_score(
        self,
        query: str,
        response: str,
        search_results: List[SearchResult]
    ) -> float:
        try:
            factors = []
            if search_results:
                avg_source = sum(r.score for r in search_results) / len(search_results)
                factors.append(avg_source)
            length = len(response.split())
            if 10 <= length <= 250:
                factors.append(1.0)
            elif length < 10:
                factors.append(0.3)
            else:
                factors.append(max(0.5, 1.0 - (length - 250) / 400))
            q_lang = arabic_processor.detect_language(query)
            r_lang = arabic_processor.detect_language(response)
            factors.append(1.0 if q_lang == r_lang or r_lang == 'ar' else 0.7)
            bad_terms = ["عذراً", "لم أتمكن", "خطأ", "غير متأكد"]
            factors.append(0.3 if any(bt in response for bt in bad_terms) else 1.0)
            weights = [0.4, 0.2, 0.2, 0.2]
            score = sum(f * w for f, w in zip(factors, weights))
            return min(max(score, 0.0), 1.0)
        except Exception:
            return 0.5

    @log_execution_time
    async def generate_rag_response(
        self,
        query: str,
        search_results: List[SearchResult],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> RAGResponse:
        start = time.time()
        try:
            prompt = await self._build_rag_prompt(query, search_results, conversation_history)
            response_text = await self._generate_text(prompt)
            conf = await self._calculate_confidence_score(query, response_text, search_results)
            elapsed = time.time() - start
            return RAGResponse(
                query=query,
                answer=response_text,
                sources=search_results,
                processing_time=elapsed,
                confidence_score=conf,
            )
        except Exception as e:
            elapsed = time.time() - start
            self.logger.error(f"Ollama RAG failed: {str(e)}")
            return RAGResponse(
                query=query,
                answer=f"عذراً، حدث خطأ: {str(e)}",
                sources=search_results,
                processing_time=elapsed,
                confidence_score=0.0,
            )

    # Convenience helpers similar to Gemini service
    async def summarize_text(self, text: str, max_length: int = 200) -> str:
        language = arabic_processor.detect_language(text)
        if language in ("ar", "mixed"):
            prompt = f"لخص النص التالي في {max_length} كلمة كحد أقصى:\n\n{text}\n\nالملخص:"
        else:
            prompt = f"Summarize the following text in maximum {max_length} words in Arabic:\n\n{text}\n\nSummary:"
        return await self._generate_text(prompt)

    async def generate_questions(self, text: str, num_questions: int = 3) -> List[str]:
        prompt = f"""
بناءً على النص التعليمي التالي، أنشئ {num_questions} أسئلة تعليمية مفيدة:

{text}

الأسئلة:
"""
        resp = await self._generate_text(prompt)
        qs: List[str] = []
        for line in resp.split('\n'):
            s = line.strip()
            if not s:
                continue
            if s.startswith('- ') or s.startswith('• ') or any(s.startswith(f"{i}.") for i in range(1, 11)):
                qs.append(s.lstrip('- •0123456789. ').strip())
        return qs[:num_questions]

    async def evaluate_answer(self, question: str, student_answer: str, correct_answer: str) -> Dict[str, Any]:
        prompt = f"""
قم بتقييم إجابة الطالب مقارنة بالإجابة الصحيحة وأعط تقييماً مفصلاً:

السؤال: {question}

إجابة الطالب: {student_answer}

الإجابة الصحيحة: {correct_answer}

يرجى تقديم التقييم في الشكل التالي:
- الدرجة: (من 0 إلى 10)
- النقاط الصحيحة:
- النقاط المفقودة:
- اقتراحات للتحسين:

التقييم:
"""
        evaluation = await self._generate_text(prompt)
        score = 5
        try:
            import re
            for line in evaluation.split('\n'):
                if 'الدرجة' in line or 'Score' in line:
                    nums = re.findall(r'\d+', line)
                    if nums:
                        score = min(10, max(0, int(nums[0])))
                        break
        except Exception:
            pass
        return {
            "score": score,
            "max_score": 10,
            "evaluation": evaluation,
            "percentage": (score / 10) * 100,
        }

    async def health_check(self) -> bool:
        try:
            await self.initialize()
            return True
        except Exception:
            return False

    async def get_service_info(self) -> Dict[str, Any]:
        healthy = await self.health_check()
        return {
            "service": "Ollama LLM Service",
            "base_url": self.base_url,
            "model": self.model,
            "healthy": healthy,
        }


# Global instance
llm_service = OllamaService()
try:
    import gc
    resource_manager.register("llm", lambda: gc.collect())
except Exception:
    pass


