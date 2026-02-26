"""Test script for web search engine fixes."""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.web_search import (
    WebSearchEngine,
    SearchResult,
    SearchEngine,
    SearchEngineManager,
    SearchResultCache,
    get_search_engine,
)


async def test_search_result_creation():
    """Test SearchResult creation and serialization."""
    print("Testing SearchResult creation...")
    
    result = SearchResult(
        id="test123",
        title="Test Title",
        snippet="Test snippet content",
        url="https://example.com/test",
        source="Test Source",
        engine=SearchEngine.BING_CN,
        rank=1,
        language="zh",
    )
    
    # Test quality score calculation
    assert 0.0 <= result.quality_score <= 1.0, "Quality score should be between 0 and 1"
    
    # Test serialization
    data = result.to_dict()
    assert data["id"] == "test123"
    assert data["title"] == "Test Title"
    assert data["engine"] == "BING_CN"
    
    # Test deserialization
    result2 = SearchResult.from_dict(data)
    assert result2.id == result.id
    assert result2.title == result.title
    assert result2.engine == result.engine
    
    print("[PASS] SearchResult creation tests passed")


async def test_search_engine_manager():
    """Test SearchEngineManager functionality."""
    print("Testing SearchEngineManager...")
    
    manager = SearchEngineManager()
    
    # Test engine availability
    engines = manager.get_available_engines()
    assert len(engines) > 0, "Should have at least one available engine"
    
    # Test priority order
    priority = manager.get_priority_order()
    assert len(priority) > 0, "Should have priority order"
    
    # Test health tracking
    initial_health = manager.get_health(SearchEngine.BING_CN)
    assert initial_health >= 0.0 and initial_health <= 1.0
    
    # Record success
    await manager.record_success(SearchEngine.BING_CN)
    health_after_success = manager.get_health(SearchEngine.BING_CN)
    assert health_after_success >= initial_health
    
    # Record failure
    await manager.record_failure(SearchEngine.BING_CN, "Test error")
    health_after_failure = manager.get_health(SearchEngine.BING_CN)
    assert health_after_failure <= health_after_success
    
    print("[PASS] SearchEngineManager tests passed")


async def test_cache_operations():
    """Test SearchResultCache functionality."""
    print("Testing SearchResultCache...")
    
    cache = SearchResultCache(ttl_hours=1)
    
    # Create test results
    results = [
        SearchResult(
            id=f"test{i}",
            title=f"Test {i}",
            snippet=f"Snippet {i}",
            url=f"https://example.com/{i}",
            source="Test",
            engine=SearchEngine.BING_CN,
            rank=i,
        )
        for i in range(3)
    ]
    
    # Test caching
    await cache.set("test query", [SearchEngine.BING_CN], 10, results)
    
    # Test retrieval
    cached = await cache.get("test query", [SearchEngine.BING_CN], 10)
    assert cached is not None, "Should retrieve cached results"
    assert len(cached) == 3, "Should have 3 cached results"
    
    # Test cache miss
    missing = await cache.get("nonexistent query", [SearchEngine.BING_CN], 10)
    assert missing is None, "Should return None for missing cache"
    
    # Test clear
    await cache.clear()
    cleared = await cache.get("test query", [SearchEngine.BING_CN], 10)
    assert cleared is None, "Cache should be cleared"
    
    print("[PASS] SearchResultCache tests passed")


async def test_quality_score_calculation():
    """Test quality score calculation."""
    print("Testing quality score calculation...")
    
    search_engine = get_search_engine()
    
    # Test with high quality result
    result = SearchResult(
        id="test1",
        title="A 股政策分析",  # Contains query terms
        snippet="A 股市场政策分析报道",
        url="https://eastmoney.com/article/123",  # High quality domain
        source="EastMoney",
        engine=SearchEngine.BING_CN,
        rank=0,  # Top rank
    )
    
    score = search_engine._calculate_quality_score(result, "A 股 政策")
    assert 0.0 <= score <= 1.0, "Score should be normalized"
    assert score > 0.5, "High quality result should have good score"
    
    # Test with low quality result
    result2 = SearchResult(
        id="test2",
        title="Random content",  # No query terms
        snippet="Unrelated snippet",
        url="https://unknown-site.com/page",
        source="Unknown",
        engine=SearchEngine.BING_CN,
        rank=10,  # Low rank
    )
    
    score2 = search_engine._calculate_quality_score(result2, "A 股 政策")
    assert 0.0 <= score2 <= 1.0, "Score should be normalized"
    assert score2 < score, "Low quality result should have lower score"
    
    # Test with empty query (edge case)
    score3 = search_engine._calculate_quality_score(result, "")
    assert 0.0 <= score3 <= 1.0, "Score should handle empty query"
    
    print("[PASS] Quality score calculation tests passed")


async def test_html_stripping():
    """Test HTML tag stripping."""
    print("Testing HTML stripping...")
    
    search_engine = get_search_engine()
    
    # Test with HTML content
    html_text = "<div><h2>Title</h2><p>Content with <b>bold</b> text</p></div>"
    clean = search_engine._strip_html_tags(html_text)
    assert "<" not in clean, "Should remove all HTML tags"
    assert ">" not in clean, "Should remove all HTML tags"
    assert "Title" in clean, "Should preserve text content"
    assert "Content" in clean, "Should preserve text content"
    
    # Test with empty string
    assert search_engine._strip_html_tags("") == ""
    
    # Test with HTML entities
    html_entities = "<p>Test &amp; example</p>"
    clean2 = search_engine._strip_html_tags(html_entities)
    assert "Test & example" in clean2, "Should decode HTML entities"
    
    print("[PASS] HTML stripping tests passed")


async def test_deduplication():
    """Test result deduplication."""
    print("Testing deduplication...")
    
    search_engine = get_search_engine()
    
    # Create duplicate results
    results = [
        SearchResult(
            id="test1",
            title="Test Title",
            snippet="Test snippet",
            url="https://example.com/1",
            source="Test",
            engine=SearchEngine.BING_CN,
            rank=0,
        ),
        SearchResult(
            id="test2",
            title="Test Title",
            snippet="Test snippet",
            url="https://example.com/1",  # Same URL
            source="Test",
            engine=SearchEngine.BAIDU,
            rank=1,
        ),
        SearchResult(
            id="test3",
            title="Different",
            snippet="Different snippet",
            url="https://example.com/2",
            source="Test",
            engine=SearchEngine.BING_CN,
            rank=2,
        ),
    ]
    
    # Deduplicate
    unique = await search_engine._deduplicate_results(results)
    assert len(unique) == 2, "Should remove duplicate URL"
    
    print("[PASS] Deduplication tests passed")


async def test_engine_initialization():
    """Test search engine initialization."""
    print("Testing engine initialization...")
    
    # Test global instance
    engine1 = get_search_engine()
    engine2 = get_search_engine()
    assert engine1 is engine2, "Should return same instance"
    
    # Test properties
    assert engine1.max_concurrent > 0, "Should have concurrent limit"
    assert engine1.cache is not None, "Should have cache"
    assert engine1.engine_manager is not None, "Should have engine manager"
    
    print("[PASS] Engine initialization tests passed")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Web Search Engine Tests")
    print("=" * 60)
    print()
    
    try:
        await test_search_result_creation()
        await test_search_engine_manager()
        await test_cache_operations()
        await test_quality_score_calculation()
        await test_html_stripping()
        await test_deduplication()
        await test_engine_initialization()
        
        print()
        print("=" * 60)
        print("All tests passed!")
        print("=" * 60)
        return 0
    except Exception as e:
        print()
        print("=" * 60)
        print(f"Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
