# test_gemini_service.py
"""
Comprehensive test suite for Gemini Service
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.gemini_service import GeminiService
from models.schemas import SearchResult, RAGResponse


class TestGeminiService:
    """Test suite for GeminiService"""
    
    @pytest.fixture
    def gemini_service(self):
        """Create a GeminiService instance for testing"""
        service = GeminiService()
        return service
    
    @pytest.fixture
    def sample_search_results(self):
        """Sample search results for testing"""
        return [
            SearchResult(
                text="الخلية هي الوحدة الأساسية للحياة",
                score=0.9,
                source="كتاب الأحياء",
                metadata={"chapter": "الخلية"}
            ),
            SearchResult(
                text="تحتوي الخلية على النواة والسيتوبلازم",
                score=0.8,
                source="مراجعة الأحياء",
                metadata={"topic": "مكونات الخلية"}
            )
        ]
    
    @pytest.mark.asyncio
    async def test_initialization(self, gemini_service):
        """Test service initialization"""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel') as mock_model:
                mock_model.return_value.generate_content.return_value.candidates = [
                    MagicMock(content=MagicMock(parts=[MagicMock(text="مرحبا")]))
                ]
                
                await gemini_service.initialize()
                assert gemini_service._initialized is True
    
    @pytest.mark.asyncio
    async def test_health_check(self, gemini_service):
        """Test health check functionality"""
        with patch.object(gemini_service, '_generate_text') as mock_generate:
            mock_generate.return_value = "صحي"
            gemini_service._initialized = True
            
            result = await gemini_service.health_check()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_rag_response_generation(self, gemini_service, sample_search_results):
        """Test RAG response generation"""
        with patch.object(gemini_service, '_generate_text') as mock_generate:
            mock_generate.return_value = "الخلية هي الوحدة الأساسية للحياة في جميع الكائنات الحية"
            gemini_service._initialized = True
            
            query = "ما هي الخلية؟"
            response = await gemini_service.generate_rag_response(query, sample_search_results)
            
            assert isinstance(response, RAGResponse)
            assert response.query == query
            assert response.answer is not None
            assert response.confidence_score >= 0.0
            assert response.processing_time >= 0.0
    
    @pytest.mark.asyncio
    async def test_text_summarization(self, gemini_service):
        """Test text summarization"""
        with patch.object(gemini_service, '_generate_text') as mock_generate:
            mock_generate.return_value = "ملخص النص"
            gemini_service._initialized = True
            
            text = "نص طويل يحتاج إلى تلخيص"
            summary = await gemini_service.summarize_text(text, max_length=50)
            
            assert summary == "ملخص النص"
    
    @pytest.mark.asyncio
    async def test_question_generation(self, gemini_service):
        """Test question generation"""
        with patch.object(gemini_service, '_generate_text') as mock_generate:
            mock_generate.return_value = "1. ما هي الخلية؟\n2. ما وظيفة النواة؟\n3. كيف تتكاثر الخلايا؟"
            gemini_service._initialized = True
            
            text = "الخلية هي الوحدة الأساسية للحياة"
            questions = await gemini_service.generate_questions(text, num_questions=3)
            
            assert len(questions) <= 3
            assert all(isinstance(q, str) for q in questions)
    
    @pytest.mark.asyncio
    async def test_concept_explanation(self, gemini_service):
        """Test concept explanation"""
        with patch.object(gemini_service, '_generate_text') as mock_generate:
            mock_generate.return_value = "شرح مفصل للمفهوم"
            gemini_service._initialized = True
            
            concept = "التنفس الخلوي"
            explanation = await gemini_service.explain_concept(concept)
            
            assert explanation == "شرح مفصل للمفهوم"
    
    @pytest.mark.asyncio
    async def test_translation(self, gemini_service):
        """Test text translation"""
        with patch.object(gemini_service, '_generate_text') as mock_generate:
            mock_generate.return_value = "النص المترجم"
            gemini_service._initialized = True
            
            text = "Text to translate"
            translation = await gemini_service.translate_text(text, "ar")
            
            assert translation == "النص المترجم"
    
    @pytest.mark.asyncio
    async def test_answer_evaluation(self, gemini_service):
        """Test answer evaluation"""
        with patch.object(gemini_service, '_generate_text') as mock_generate:
            mock_generate.return_value = "الدرجة: 8\nتقييم جيد"
            gemini_service._initialized = True
            
            question = "ما هي الخلية؟"
            student_answer = "الوحدة الأساسية للحياة"
            correct_answer = "الخلية هي الوحدة الأساسية للحياة"
            
            evaluation = await gemini_service.evaluate_answer(question, student_answer, correct_answer)
            
            assert isinstance(evaluation, dict)
            assert "score" in evaluation
            assert "evaluation" in evaluation
            assert 0 <= evaluation["score"] <= 10
    
    @pytest.mark.asyncio
    async def test_error_handling(self, gemini_service, sample_search_results):
        """Test error handling in RAG response generation"""
        with patch.object(gemini_service, '_generate_text') as mock_generate:
            mock_generate.side_effect = Exception("API Error")
            gemini_service._initialized = True
            
            query = "سؤال اختبار"
            response = await gemini_service.generate_rag_response(query, sample_search_results)
            
            assert isinstance(response, RAGResponse)
            assert "خطأ" in response.answer
            assert response.confidence_score == 0.0
    
    @pytest.mark.asyncio
    async def test_confidence_calculation(self, gemini_service, sample_search_results):
        """Test confidence score calculation"""
        gemini_service._initialized = True
        
        query = "ما هي الخلية؟"
        response = "الخلية هي الوحدة الأساسية للحياة"
        
        confidence = await gemini_service._calculate_confidence_score(query, response, sample_search_results)
        
        assert 0.0 <= confidence <= 1.0


# Integration tests
class TestGeminiServiceIntegration:
    """Integration tests for GeminiService"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete workflow with real API (requires API key)"""
        service = GeminiService()
        
        # Skip if no API key
        if not os.getenv("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")
        
        try:
            await service.initialize()
            
            # Test health check
            health = await service.health_check()
            assert health is True
            
            # Test simple generation
            response = await service._generate_text("مرحبا")
            assert len(response) > 0
            
        except Exception as e:
            pytest.fail(f"Integration test failed: {str(e)}")


# Performance tests
class TestGeminiServicePerformance:
    """Performance tests for GeminiService"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, gemini_service, sample_search_results):
        """Test handling of concurrent requests"""
        with patch.object(gemini_service, '_generate_text') as mock_generate:
            mock_generate.return_value = "إجابة اختبار"
            gemini_service._initialized = True
            
            queries = [f"سؤال {i}" for i in range(10)]
            
            start_time = asyncio.get_event_loop().time()
            
            tasks = [
                gemini_service.generate_rag_response(query, sample_search_results)
                for query in queries
            ]
            
            responses = await asyncio.gather(*tasks)
            
            end_time = asyncio.get_event_loop().time()
            
            assert len(responses) == 10
            assert all(isinstance(r, RAGResponse) for r in responses)
            
            # Should complete within reasonable time
            total_time = end_time - start_time
            assert total_time < 10.0  # 10 seconds max
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage(self, gemini_service):
        """Test memory usage doesn't grow excessively"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        with patch.object(gemini_service, '_generate_text') as mock_generate:
            mock_generate.return_value = "إجابة" * 100  # Long response
            gemini_service._initialized = True
            
            # Generate many responses
            for i in range(50):
                await gemini_service._generate_text(f"سؤال {i}")
                if i % 10 == 0:
                    gc.collect()
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 100MB)
            assert memory_increase < 100 * 1024 * 1024


# Fixtures for pytest
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Custom markers
pytest_plugins = []

def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


# Run tests command examples
"""
# Run all tests
pytest test_gemini_service.py -v

# Run only unit tests (exclude integration and performance)
pytest test_gemini_service.py -v -m "not integration and not performance"

# Run integration tests only
pytest test_gemini_service.py -v -m integration

# Run performance tests only
pytest test_gemini_service.py -v -m performance

# Run with coverage
pytest test_gemini_service.py --cov=services.gemini_service --cov-report=html

# Run specific test class
pytest test_gemini_service.py::TestGeminiService -v

# Run specific test method
pytest test_gemini_service.py::TestGeminiService::test_initialization -v
"""