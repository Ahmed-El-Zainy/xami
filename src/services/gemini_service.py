# services/gemini_service.py
import asyncio
import time
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json
import os 
import sys 


SCR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(SCR_DIR)))

from src.config.config_settings import settings
from src.config.logging_config import get_logger, log_execution_time, CustomLoggerTracker
from src.models.schemas import SearchResult, RAGResponse
from src.utilities.arabic_utils import arabic_processor


try:
    # from logger.custom_logger import CustomLoggerTracker
    custom = CustomLoggerTracker()
    logger = custom.get_logger("gemini_serivce")
    logger.info("Custom Logger Start Working.....")

except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("gemini_service")
    logger.info("Using standard logger - custom logger not available")


class GeminiService:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.model = None
        self.generation_config = None
        self.safety_settings = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize Gemini API"""
        if self._initialized:
            return
            
        try:
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not provided in settings")
            
            self.logger.info(f"Initializing Gemini API with model: {settings.GEMINI_MODEL}")
            
            # Configure the API
            genai.configure(api_key=settings.GEMINI_API_KEY)
            
            # Initialize the model
            self.model = genai.GenerativeModel(
                model_name=settings.GEMINI_MODEL,
                generation_config=self._get_generation_config(),
                safety_settings=self._get_safety_settings()
            )
            
            # Test the model
            test_response = await self._generate_text("مرحبا")
            self.logger.info("Gemini API initialized successfully")
            self._initialized = True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini API: {str(e)}")
            raise
    
    def _get_generation_config(self):
        """Get generation configuration for Gemini"""
        return genai.types.GenerationConfig(
            candidate_count=1,
            temperature=0.7,
            top_p=0.8,
            top_k=40,
            max_output_tokens=2048,
        )
    
    def _get_safety_settings(self):
        """Get safety settings for Gemini"""
        return [
            {
                "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            },
            {
                "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            },
            {
                "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            },
            {
                "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            },
        ]
    
    @log_execution_time
    async def generate_rag_response(
        self, 
        query: str, 
        search_results: List[SearchResult],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> RAGResponse:
        """Generate RAG response using Gemini"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Generating RAG response for query: {query[:50]}...")
            
            # Build the prompt
            prompt = await self._build_rag_prompt(query, search_results, conversation_history)
            
            # Generate response
            response_text = await self._generate_text(prompt)
            
            # Calculate confidence score
            confidence_score = await self._calculate_confidence_score(
                query, response_text, search_results
            )
            
            processing_time = time.time() - start_time
            
            # Create RAG response
            rag_response = RAGResponse(
                query=query,
                answer=response_text,
                sources=search_results,
                processing_time=processing_time,
                confidence_score=confidence_score
            )
            
            self.logger.info(f"RAG response generated successfully in {processing_time:.2f}s")
            return rag_response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error generating RAG response: {str(e)}")
            
            # Return error response
            return RAGResponse(
                query=query,
                answer=f"عذراً، حدث خطأ في معالجة سؤالك: {str(e)}",
                sources=search_results,
                processing_time=processing_time,
                confidence_score=0.0
            )
    
    async def _build_rag_prompt(
        self, 
        query: str, 
        search_results: List[SearchResult],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Build the RAG prompt for Gemini"""
        
        # Detect query language
        query_language = arabic_processor.detect_language(query)
        
        # Build context from search results
        context_parts = []
        for i, result in enumerate(search_results[:5], 1):  # Use top 5 results
            context_parts.append(f"[مصدر {i}]\n{result.text}\n")
        
        context = "\n".join(context_parts)
        
        # Build conversation history
        history_text = ""
        if conversation_history:
            history_parts = []
            for turn in conversation_history[-3:]:  # Last 3 turns
                history_parts.append(f"المستخدم: {turn.get('user', '')}")
                history_parts.append(f"المساعد: {turn.get('assistant', '')}")
            history_text = "\n".join(history_parts)
        
        # Choose prompt based on language
        if query_language == "ar" or query_language == "mixed":
            prompt = f"""أنت مساعد ذكي متخصص في التعليم الثانوي العربي. مهمتك هي الإجابة على أسئلة الطلاب بناءً على المحتوى التعليمي المقدم.

{'## سياق المحادثة السابقة:' + chr(10) + history_text + chr(10) if history_text else ''}

## المحتوى التعليمي المرجعي:
{context}

## سؤال الطالب:
{query}

## تعليمات الإجابة:
1. اجب بوضوح ودقة باللغة العربية
2. استخدم المحتوى المرجعي المقدم أعلاه فقط
3. إذا لم تجد الإجابة في المحتوى المرجعي، اذكر ذلك بوضوح
4. قدم أمثلة عملية عند الإمكان
5. استخدم تنسيقاً واضحاً مع نقاط أو ترقيم عند الحاجة
6. تأكد من دقة المعلومات العلمية والرياضية

## الإجابة:"""
        else:
            prompt = f"""You are an intelligent assistant specialized in Arabic secondary education. Your task is to answer student questions based on the provided educational content.

{'## Previous Conversation Context:' + chr(10) + history_text + chr(10) if history_text else ''}

## Reference Educational Content:
{context}

## Student Question:
{query}

## Answer Instructions:
1. Answer clearly and accurately in Arabic
2. Use only the reference content provided above
3. If you cannot find the answer in the reference content, state this clearly
4. Provide practical examples when possible
5. Use clear formatting with bullet points or numbering when needed
6. Ensure accuracy of scientific and mathematical information

## Answer:"""
        
        return prompt
    
    async def _generate_text(self, prompt: str) -> str:
        """Generate text using Gemini"""
        try:
            # Create async wrapper for Gemini API
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None, 
                lambda: self.model.generate_content(prompt)
            )
            
            if response.candidates and response.candidates[0].content:
                return response.candidates[0].content.parts[0].text
            else:
                return "عذراً، لم أتمكن من توليد إجابة مناسبة لسؤالك."
                
        except Exception as e:
            self.logger.error(f"Error generating text with Gemini: {str(e)}")
            return f"عذراً، حدث خطأ في توليد الإجابة: {str(e)}"
    
    async def _calculate_confidence_score(
        self, 
        query: str, 
        response: str, 
        search_results: List[SearchResult]
    ) -> float:
        """Calculate confidence score for the response"""
        try:
            confidence_factors = []
            
            # 1. Source quality (average of search result scores)
            if search_results:
                avg_source_score = sum(r.score for r in search_results) / len(search_results)
                confidence_factors.append(avg_source_score)
            
            # 2. Response length appropriateness
            response_length = len(response.split())
            if 10 <= response_length <= 200:  # Reasonable length
                length_score = 1.0
            elif response_length < 10:
                length_score = 0.3  # Too short
            else:
                length_score = max(0.5, 1.0 - (response_length - 200) / 300)  # Too long
            confidence_factors.append(length_score)
            
            # 3. Language consistency
            query_lang = arabic_processor.detect_language(query)
            response_lang = arabic_processor.detect_language(response)
            lang_consistency = 1.0 if query_lang == response_lang else 0.7
            confidence_factors.append(lang_consistency)
            
            # 4. Error indicators
            error_indicators = ["عذراً", "خطأ", "لم أتمكن", "غير متأكد"]
            has_error = any(indicator in response for indicator in error_indicators)
            error_score = 0.3 if has_error else 1.0
            confidence_factors.append(error_score)
            
            # Calculate weighted average
            weights = [0.4, 0.2, 0.2, 0.2]  # Source quality is most important
            confidence_score = sum(f * w for f, w in zip(confidence_factors, weights))
            
            return min(max(confidence_score, 0.0), 1.0)  # Clamp to [0, 1]
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence score: {str(e)}")
            return 0.5  # Default moderate confidence
    
    async def summarize_text(self, text: str, max_length: int = 200) -> str:
        """Summarize text using Gemini"""
        if not self._initialized:
            await self.initialize()
        
        try:
            language = arabic_processor.detect_language(text)
            
            if language == "ar" or language == "mixed":
                prompt = f"""لخص النص التالي في {max_length} كلمة كحد أقصى:

{text}

الملخص:"""
            else:
                prompt = f"""Summarize the following text in maximum {max_length} words in Arabic:

{text}

Summary:"""
            
            summary = await self._generate_text(prompt)
            return summary
            
        except Exception as e:
            self.logger.error(f"Error summarizing text: {str(e)}")
            return text[:max_length] + "..." if len(text) > max_length else text
    
    async def generate_questions(self, text: str, num_questions: int = 3) -> List[str]:
        """Generate questions based on text content"""
        if not self._initialized:
            await self.initialize()
        
        try:
            prompt = f"""بناءً على النص التعليمي التالي، أنشئ {num_questions} أسئلة تعليمية مفيدة:

{text}

الأسئلة:"""
            
            response = await self._generate_text(prompt)
            
            # Parse questions from response
            questions = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line.startswith('- ') or line.startswith('• ') or 
                           any(line.startswith(f'{i}.') for i in range(1, 11))):
                    # Clean the question
                    question = line.lstrip('- • 0123456789. ').strip()
                    if question:
                        questions.append(question)
            
            return questions[:num_questions]
            
        except Exception as e:
            self.logger.error(f"Error generating questions: {str(e)}")
            return []
    
    async def explain_concept(self, concept: str, context: str = "") -> str:
        """Explain a concept in simple terms"""
        if not self._initialized:
            await self.initialize()
        
        try:
            prompt = f"""اشرح المفهوم التالي بطريقة بسيطة ومفهومة لطالب في المرحلة الثانوية:

المفهوم: {concept}

{f'السياق: {context}' if context else ''}

الشرح:"""
            
            explanation = await self._generate_text(prompt)
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error explaining concept: {str(e)}")
            return f"عذراً، لم أتمكن من شرح المفهوم: {concept}"
    
    async def translate_text(self, text: str, target_language: str = "ar") -> str:
        """Translate text to target language"""
        if not self._initialized:
            await self.initialize()
        
        try:
            if target_language == "ar":
                prompt = f"""ترجم النص التالي إلى اللغة العربية بدقة مع الحفاظ على المعنى العلمي:

{text}

الترجمة:"""
            else:
                prompt = f"""Translate the following Arabic text to {target_language} accurately while maintaining scientific meaning:

{text}

Translation:"""
            
            translation = await self._generate_text(prompt)
            return translation
            
        except Exception as e:
            self.logger.error(f"Error translating text: {str(e)}")
            return text
    
    async def evaluate_answer(self, question: str, student_answer: str, correct_answer: str) -> Dict[str, Any]:
        """Evaluate student answer against correct answer"""
        if not self._initialized:
            await self.initialize()
        
        try:
            prompt = f"""قم بتقييم إجابة الطالب مقارنة بالإجابة الصحيحة وأعط تقييماً مفصلاً:

السؤال: {question}

إجابة الطالب: {student_answer}

الإجابة الصحيحة: {correct_answer}

يرجى تقديم التقييم في الشكل التالي:
- الدرجة: (من 0 إلى 10)
- النقاط الصحيحة:
- النقاط المفقودة:
- اقتراحات للتحسين:

التقييم:"""
            
            evaluation = await self._generate_text(prompt)
            
            # Try to parse score from response
            score = 5  # default score
            lines = evaluation.split('\n')
            for line in lines:
                if 'الدرجة' in line or 'Score' in line:
                    # Try to extract number
                    import re
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        score = min(10, max(0, int(numbers[0])))
                        break
            
            return {
                "score": score,
                "max_score": 10,
                "evaluation": evaluation,
                "percentage": (score / 10) * 100
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating answer: {str(e)}")
            return {
                "score": 0,
                "max_score": 10,
                "evaluation": f"خطأ في التقييم: {str(e)}",
                "percentage": 0
            }
    
    async def health_check(self) -> bool:
        """Check if Gemini API is working"""
        try:
            if not self._initialized:
                await self.initialize()
            
            test_response = await self._generate_text("مرحبا")
            return len(test_response) > 0
            
        except Exception as e:
            self.logger.error(f"Gemini health check failed: {str(e)}")
            return False
    
    async def get_service_info(self) -> Dict[str, Any]:
        """Get service information and status"""
        try:
            is_healthy = await self.health_check()
            
            return {
                "service": "Gemini Service",
                "model": settings.GEMINI_MODEL if hasattr(settings, 'GEMINI_MODEL') else "Unknown",
                "initialized": self._initialized,
                "healthy": is_healthy,
                "capabilities": [
                    "RAG Response Generation",
                    "Text Summarization", 
                    "Question Generation",
                    "Concept Explanation",
                    "Text Translation",
                    "Answer Evaluation"
                ]
            }
        except Exception as e:
            return {
                "service": "Gemini Service",
                "error": str(e),
                "healthy": False
            }


# Global Gemini service instance
gemini_service = GeminiService()


# Test Sample and Usage Examples
async def test_gemini_service():
    """Comprehensive test suite for Gemini Service"""
    print("=" * 60)
    print("🚀 Testing Gemini Service")
    print("=" * 60)
    
    try:
        # Initialize service
        print("1. Initializing Gemini Service...")
        await gemini_service.initialize()
        print("   ✅ Service initialized successfully")
        
        # Test health check
        print("\n2. Testing health check...")
        health = await gemini_service.health_check()
        print(f"   ✅ Health check: {'Passed' if health else 'Failed'}")
        
        # Test service info
        print("\n3. Getting service information...")
        info = await gemini_service.get_service_info()
        print(f"   ✅ Service Info:")
        for key, value in info.items():
            print(f"      {key}: {value}")
        
        # Create sample search results
        sample_search_results = [
            SearchResult(
                text="الخلية هي الوحدة الأساسية للحياة في جميع الكائنات الحية. تحتوي الخلية على النواة التي تحمل المادة الوراثية DNA.",
                score=0.9,
                source="كتاب الأحياء - الصف الثاني الثانوي",
                metadata={"chapter": "الخلية", "page": 15}
            ),
            SearchResult(
                text="تنقسم الخلايا إلى نوعين رئيسيين: الخلايا بدائية النواة والخلايا حقيقية النواة. الخلايا حقيقية النواة تحتوي على نواة محاطة بغشاء.",
                score=0.85,
                source="مراجعة الأحياء",
                metadata={"topic": "أنواع الخلايا"}
            )
        ]
        
        # Test RAG response generation
        print("\n4. Testing RAG response generation...")
        query = "ما هي الخلية وما أنواعها؟"
        rag_response = await gemini_service.generate_rag_response(query, sample_search_results)
        print(f"   ✅ RAG Response generated:")
        print(f"      Query: {rag_response.query}")
        print(f"      Answer: {rag_response.answer[:100]}...")
        print(f"      Confidence: {rag_response.confidence_score:.2f}")
        print(f"      Processing Time: {rag_response.processing_time:.2f}s")
        
        # Test with conversation history
        print("\n5. Testing RAG with conversation history...")
        conversation_history = [
            {"user": "ما هي الخلية؟", "assistant": "الخلية هي الوحدة الأساسية للحياة"},
            {"user": "أريد معرفة المزيد", "assistant": "يمكنني شرح أنواع الخلايا"}
        ]
        rag_response_with_history = await gemini_service.generate_rag_response(
            "اشرح لي الفرق بين أنواع الخلايا", 
            sample_search_results, 
            conversation_history
        )
        print(f"   ✅ RAG with history: {rag_response_with_history.answer[:100]}...")
        
        # Test text summarization
        print("\n6. Testing text summarization...")
        long_text = """الخلية هي الوحدة الأساسية للحياة في جميع الكائنات الحية. تحتوي الخلية على مكونات مختلفة مثل النواة والسيتوبلازم والغشاء الخلوي. النواة تحمل المادة الوراثية DNA التي تحدد صفات الكائن الحي. السيتوبلازم هو المادة الهلامية التي تحيط بالنواة وتحتوي على العضيات المختلفة مثل الميتوكوندريا والريبوسومات. الغشاء الخلوي يحيط بالخلية ويتحكم في دخول وخروج المواد."""
        summary = await gemini_service.summarize_text(long_text, max_length=50)
        print(f"   ✅ Summary: {summary}")
        
        # Test question generation
        print("\n7. Testing question generation...")
        educational_text = "الميتوكوندريا هي عضية خلوية تُعرف بأنها مصنع الطاقة في الخلية. تقوم الميتوكوندريا بإنتاج ATP من خلال عملية التنفس الخلوي."
        questions = await gemini_service.generate_questions(educational_text, num_questions=3)
        print(f"   ✅ Generated Questions:")
        for i, question in enumerate(questions, 1):
            print(f"      {i}. {question}")
        
        # Test concept explanation
        print("\n8. Testing concept explanation...")
        concept = "التنفس الخلوي"
        context = "عملية تحويل الجلوكوز إلى طاقة في الميتوكوندريا"
        explanation = await gemini_service.explain_concept(concept, context)
        print(f"   ✅ Concept Explanation: {explanation[:150]}...")
        
        # Test translation
        print("\n9. Testing translation...")
        english_text = "Photosynthesis is the process by which plants convert sunlight into energy."
        translation = await gemini_service.translate_text(english_text, "ar")
        print(f"   ✅ Translation: {translation}")
        
        # Test answer evaluation
        print("\n10. Testing answer evaluation...")
        question = "ما هي وظيفة الميتوكوندريا؟"
        student_answer = "الميتوكوندريا تنتج الطاقة للخلية"
        correct_answer = "الميتوكوندريا هي عضية خلوية مسؤولة عن إنتاج الطاقة (ATP) من خلال عملية التنفس الخلوي"
        evaluation = await gemini_service.evaluate_answer(question, student_answer, correct_answer)
        print(f"   ✅ Answer Evaluation:")
        print(f"      Score: {evaluation['score']}/{evaluation['max_score']}")
        print(f"      Percentage: {evaluation['percentage']}%")
        print(f"      Evaluation: {evaluation['evaluation'][:100]}...")
        
        print("\n" + "=" * 60)
        print("🎉 All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


# Performance test
async def performance_test():
    """Test service performance with multiple concurrent requests"""
    print("\n🔥 Running Performance Test...")
    
    queries = [
        "ما هي الخلية؟",
        "اشرح عملية البناء الضوئي",
        "ما هي أنواع الأنسجة؟",
        "كيف يتم التنفس الخلوي؟",
        "ما هي وظيفة الدم؟"
    ]
    
    sample_results = [
        SearchResult(
            text="نص تعليمي متعلق بالسؤال",
            score=0.8,
            source="مصدر تعليمي",
            metadata={}
        )
    ]
    
    start_time = time.time()
    
    # Run concurrent requests
    tasks = [
        gemini_service.generate_rag_response(query, sample_results) 
        for query in queries
    ]
    
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    total_time = time.time() - start_time
    
    successful = sum(1 for r in responses if isinstance(r, RAGResponse))
    failed = len(responses) - successful
    
    print(f"Performance Results:")
    print(f"  Total requests: {len(queries)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average time per request: {total_time/len(queries):.2f}s")


if __name__ == "__main__":
    async def main():
        # Run comprehensive tests
        await test_gemini_service()
        
        # Run performance tests
        await performance_test()
    
    # Run the tests
    asyncio.run(main())