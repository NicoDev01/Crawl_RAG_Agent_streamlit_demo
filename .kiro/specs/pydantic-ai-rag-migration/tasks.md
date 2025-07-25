# Implementation Plan

- [x] 1. Setup Core Pydantic AI Agent with GoogleModel Integration



  - Replace manual Gemini API calls with native Pydantic AI GoogleModel
  - Configure agent with GoogleModel("gemini-2.5-flash") as primary model
  - Implement OpenAI fallback model configuration
  - Update agent initialization to use structured dependencies
  - _Requirements: 1.1, 3.1, 3.2, 3.3_




- [x] 2. Create Structured Data Models for RAG Pipeline

- [x] 2.1 Implement Input Data Models

  - Create QueryVariations model with validation for multi-query generation



  - Implement QueryStrategy enum for adaptive query complexity
  - Add validation rules for query variations (1-3 items, complexity scoring)
  - _Requirements: 2.1, 2.2, 2.3_


- [x] 2.2 Implement Processing Data Models

  - Create RetrievalResult model for structured retrieval outputs
  - Implement DocumentChunk and DocumentMetadata models with validation
  - Add RankedDocuments model for re-ranking results with scores
  - _Requirements: 5.2, 5.3_

- [x] 2.3 Implement Output Data Models


  - Create StructuredRagAnswer model with summary, key_details, and sources
  - Add SourceReference model for structured source citations
  - Implement RetrievalMetadata model for quality metrics and confidence scoring
  - _Requirements: 6.1, 6.2, 6.3_

- [ ] 3. Migrate RAG Functions to Pydantic AI Tools


- [x] 3.1 Implement Multi-Query Generator Tool




  - Convert existing query generation logic to @agent.tool function
  - Add adaptive strategy based on question complexity analysis
  - Implement structured QueryVariations output with validation
  - Add parallel query generation with asyncio.gather for performance
  - _Requirements: 2.1, 2.2, 2.4, 5.1_

- [x] 3.2 Implement Structured Retrieval Tool




  - Migrate retrieve function to use structured input/output models
  - Implement parallel multi-query retrieval with asyncio.gather
  - Add unified relevance filtering with structured metadata
  - Integrate embedding cache with structured query processing
  - _Requirements: 2.4, 4.1, 5.1, 5.2_




- [x] 3.3 Implement Re-Ranking Tool as Separate Component

  - Extract Vertex AI re-ranking logic into dedicated @agent.tool
  - Add structured input validation for documents and queries



  - Implement structured score output with RankedDocuments model
  - Add fallback to score-based ranking when Vertex AI unavailable
  - _Requirements: 4.3, 5.4_

- [x] 3.4 Implement Context Formatter Tool

  - Create @agent.tool for context formatting with structured inputs
  - Add validation for document chunks and metadata consistency
  - Implement structured source reference generation with clickable links
  - Add context quality scoring and metadata enrichment
  - _Requirements: 5.3, 6.2_

- [x] 4. Integrate Native Pydantic AI Caching Mechanisms

- [x] 4.1 Evaluate and Implement Pydantic AI Cache Integration




  - Test InMemoryCache and RedisCache performance vs custom caches
  - Implement cache configuration through RAGDependencies
  - Add cache hit/miss metrics and performance monitoring
  - Migrate or integrate existing LLMResponseCache and QueryEmbeddingCache
  - _Requirements: 4.1, 4.2_

- [x] 4.2 Optimize Batch Processing with Structured Models



  - Update batch processing to use structured Pydantic models
  - Implement parallel processing with structured error handling
  - Add batch size optimization based on model constraints
  - Integrate structured logging for batch operations
  - _Requirements: 4.2_

- [x] 5. Implement Advanced Error Handling and Fallback Strategies

- [x] 5.1 Create Structured Error Models



  - Implement RAGError model with error types and recovery suggestions
  - Add ErrorType enum for different failure scenarios
  - Create structured error responses with contextual information
  - _Requirements: 1.3_

- [x] 5.2 Implement Model Fallback Logic



  - Add automatic fallback from GoogleModel to OpenAI on failures
  - Implement circuit breaker pattern for model switching
  - Add structured error logging and recovery metrics
  - Test fallback scenarios with structured validation
  - _Requirements: 3.3, 1.4_

- [x] 5.3 Add Service Fallback Mechanisms



  - Implement fallback from Vertex AI re-ranking to score-based ranking
  - Add fallback from multi-query to single-query on embedding failures
  - Create graceful degradation with structured status reporting
  - _Requirements: 4.3_

- [x] 6. Implement Structured Output Formats


- [x] 6.1 Add Structured Response Generation



  - Implement agent response formatting with StructuredRagAnswer
  - Add confidence scoring based on retrieval quality metrics
  - Create structured source references with metadata
  - Add response validation and error handling
  - _Requirements: 6.1, 6.3_


- [x] 6.2 Implement Output Format Switching


  - Add capability to switch between structured and text responses
  - Implement format_structured_answer function integration
  - Add user preference handling for output formats
  - Test both output formats with validation
  - _Requirements: 6.4_

- [x] 7. Ensure Backward Compatibility and Integration

- [x] 7.1 Implement Configuration Adapter Layer



  - Create adapter to map existing CLI parameters to Pydantic models
  - Ensure all environment variables work with new structure
  - Add configuration validation and migration helpers
  - _Requirements: 7.4_

- [x] 7.2 Update Streamlit Integration



  - Modify Streamlit app to work with new structured agent
  - Test all existing UI functionality with new backend
  - Ensure no breaking changes in user interface
  - Add structured error display in UI
  - _Requirements: 7.3_

- [x] 7.3 Maintain CLI Compatibility

  - Update CLI interface to work with new agent structure
  - Ensure all existing CLI parameters function correctly
  - Add new structured output options to CLI
  - Test CLI with both old and new parameter combinations
  - _Requirements: 7.1, 7.2_

- [ ] 8. Performance Testing and Optimization
- [ ] 8.1 Benchmark New vs Old Implementation
  - Create performance comparison tests for retrieval speed
  - Measure memory usage with structured models vs manual approach
  - Test caching effectiveness with Pydantic AI vs custom caches
  - Document performance improvements or regressions
  - _Requirements: 4.1, 4.2_

- [ ] 8.2 Optimize Structured Model Performance
  - Profile Pydantic model validation overhead
  - Optimize serialization/deserialization of large document sets
  - Tune batch sizes for optimal throughput with structured models
  - Add performance monitoring and alerting
  - _Requirements: 4.2_

- [ ] 9. Comprehensive Testing and Validation
- [ ] 9.1 Implement Unit Tests for All Tools
  - Create tests for each @agent.tool function with mock contexts
  - Add Pydantic model validation tests with edge cases
  - Test error handling and fallback scenarios
  - Implement integration tests for tool interactions
  - _Requirements: 1.2, 1.3, 1.4_

- [ ] 9.2 End-to-End Integration Testing
  - Test complete RAG pipeline with structured models
  - Validate all output formats and error scenarios
  - Test backward compatibility with existing configurations
  - Perform load testing with concurrent requests
  - _Requirements: 7.1, 7.2, 7.3_

- [x] 10. Documentation and Cleanup


- [x] 10.1 Update Documentation



  - Document new structured models and their usage
  - Create migration guide from old to new implementation
  - Add examples of structured outputs and error handling
  - Update API documentation for new tool functions

  - _Requirements: All requirements for maintainability_

- [ ] 10.2 Code Cleanup and Optimization
  - Remove deprecated manual API call functions
  - Clean up unused imports and legacy code paths
  - Optimize imports and dependencies
  - Add comprehensive type hints throughout codebase
  - _Requirements: 1.1, 1.2_