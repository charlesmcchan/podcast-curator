# Implementation Plan

- [x] 1. Set up project structure and dependencies
  - Create Python project structure with src/nanook-curator directory
  - Set up pyproject.toml with uv compatibility and required dependencies (langgraph, pydantic, youtube-transcript-api, google-api-python-client, openai)
  - Initialize uv virtual environment and dependency management
  - Create basic project configuration files (.env.example, .gitignore)
  - _Requirements: All requirements depend on proper project setup_

- [x] 2. Implement core data models and state management
- [x] 2.1 Create VideoData and CuratorState models
  - Implement VideoData Pydantic model with all required fields (video_id, title, channel, view_count, like_count, comment_count, upload_date, transcript, quality_score, key_topics)
  - Implement CuratorState model with search parameters, processing state, iterative refinement state, and error handling
  - Add validation methods and type hints for all model fields
  - Create src/nanook_curator/models.py file
  - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1_

- [x] 2.2 Create configuration management system
  - Implement Configuration class for API keys, search parameters, and quality thresholds
  - Create environment variable handling for sensitive data (YouTube API key, OpenAI API key)
  - Add configuration validation and default value management
  - Create src/nanook_curator/config.py file
  - _Requirements: 5.1, 5.2_

- [x] 3. Implement YouTube API integration
- [x] 3.1 Create YouTube Data API client wrapper
  - Implement YouTube API client with authentication and rate limiting
  - Create methods for video search with date filtering and trending indicators
  - Add error handling for API rate limits and authentication failures
  - Implement retry logic with exponential backoff
  - Create src/nanook_curator/youtube_client.py file
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 3.2 Implement video discovery functionality
  - Create video search function with keyword-based discovery
  - Implement 7-day date filtering for weekly podcast focus
  - Add trending status evaluation (view count growth, engagement rate)
  - Filter videos by minimum view count (1000+ views) and age requirements
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 3.3 Create video details fetcher
  - Implement detailed metadata retrieval (view count, like count, comment count, upload date)
  - Add engagement metrics calculation (like-to-dislike ratio, view-to-subscriber ratio)
  - Create batch processing for multiple video details
  - Handle missing or restricted video data gracefully
  - _Requirements: 3.1, 3.3, 3.4_

- [x] 4. Implement transcript processing
- [x] 4.1 Create transcript fetcher using youtube-transcript-api
  - Implement transcript retrieval with fallback handling for unavailable transcripts
  - Add transcript cleaning and parsing functionality
  - Create error handling for videos without transcripts
  - Log transcript availability issues for debugging
  - Create src/nanook_curator/transcript_processor.py file
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 4.2 Implement transcript analysis
  - Create key topic extraction from transcript text
  - Implement content quality analysis (coherence, information density)
  - Add technical accuracy indicators detection
  - Extract main points and technical details from transcripts
  - _Requirements: 2.4, 3.2_

- [x] 5. Implement quality evaluation system
- [x] 5.1 Create engagement metrics analyzer
  - Implement like-to-dislike ratio calculation with 80% minimum threshold
  - Create comment sentiment analysis functionality
  - Add view-to-subscriber ratio evaluation
  - Combine engagement metrics into overall engagement score
  - _Requirements: 3.1, 3.3_

- [x] 5.2 Implement content quality scoring
  - Create transcript coherence evaluation algorithm
  - Implement information density measurement
  - Add technical accuracy indicators assessment
  - Combine content metrics with engagement scores for final quality score
  - _Requirements: 3.2, 3.4_

- [x] 5.3 Create video ranking system
  - Implement combined quality scoring (engagement + content + freshness)
  - Add quality threshold evaluation (minimum 3 videos above 70% threshold)
  - Create ranking algorithm to select top 3-5 videos
  - Implement quality assessment feedback for search refinement
  - _Requirements: 3.4, 3.5_

- [ ] 6. Implement search refinement system
- [x] 6.1 Create iterative search parameter refinement
  - Implement progressive search strategy with 3 refinement attempts
  - Add keyword expansion with synonyms and related terms
  - Create date range expansion logic (7 days → 14 days as fallback)
  - Implement broader search terms for final attempt
  - _Requirements: 1.4, 3.5_

- [x] 6.2 Add quality-based search refinement
  - Implement refinement triggers when quality threshold not met
  - Create search term adjustment based on quality failure analysis
  - Add logic to return to discovery with expanded parameters
  - Ensure weekly focus is maintained during refinement
  - _Requirements: 3.5, 1.3_

- [ ] 7. Implement podcast script generation
- [x] 7.1 Create OpenAI integration for script generation
  - Implement OpenAI API client with authentication
  - Create prompt templates for podcast script generation
  - Add error handling and retry logic for API failures
  - Implement response parsing and validation
  - _Requirements: 4.1, 4.2_

- [ ] 7.2 Implement script synthesis and structuring
  - Create script generation from top 3-5 ranked videos
  - Implement coherent narrative flow with introduction, main content, and conclusion
  - Add smooth transitions between topics and source attribution
  - Ensure 750-1500 word target (5-10 minutes speaking time)
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 7.3 Add automatic script length management
  - Implement content trimming when script exceeds 10-minute target
  - Create importance-based section prioritization for trimming
  - Maintain script coherence during automatic editing
  - Add word count and estimated duration tracking
  - _Requirements: 4.5_

- [ ] 8. Implement LangGraph workflow orchestration
- [ ] 8.1 Create LangGraph nodes for each processing step
  - Implement discover_videos_node with state management
  - Create fetch_video_details_node and fetch_transcripts_node for parallel processing
  - Implement evaluate_quality_node and rank_videos_node
  - Create generate_script_node and store_results_node
  - Add refine_search_node for iterative improvement
  - Create src/nanook_curator/workflow.py file
  - _Requirements: All requirements integrated through workflow_

- [ ] 8.2 Implement LangGraph state flow and conditional routing
  - Create StateGraph with CuratorState schema
  - Implement conditional edges for discovery success/failure routing
  - Add parallel processing edges for video details and transcript fetching
  - Create quality-based conditional routing for refinement loops
  - Add error handling decorators for all nodes
  - _Requirements: 5.4, 3.5_

- [ ] 8.3 Add workflow error handling and recovery
  - Implement graceful degradation for missing transcripts and API failures
  - Create comprehensive error logging to state.errors list with detailed context
  - Add retry mechanisms with exponential backoff
  - Ensure non-blocking errors allow process continuation with remaining videos
  - _Requirements: 5.3_

- [ ] 9. Implement data persistence and storage
- [ ] 9.1 Create results storage system
  - Implement podcast script storage with metadata
  - Create storage for video source information and quality scores
  - Add generation timestamp and processing time tracking
  - Store search refinement history for analysis
  - _Requirements: 5.2, 5.3_

- [ ] 9.2 Add historical data management and script comparison
  - Implement script history maintenance with performance metrics
  - Create ScriptHistory model for storing multiple script versions
  - Add script comparison functionality based on quality metrics
  - Implement script selection algorithm for choosing best script from multiple options
  - Add data cleanup for old entries based on retention policy
  - _Requirements: 5.2, 5.4_

- [ ] 10. Create CLI interface and configuration
- [ ] 10.1 Implement command-line interface
  - Create CLI commands for manual execution and configuration
  - Add options for one-time runs vs scheduled execution
  - Implement configuration file management through CLI
  - Add verbose logging and debug options
  - _Requirements: 5.1_

- [ ] 10.2 Add testing and validation commands
  - Create CLI commands for testing API connections
  - Implement dry-run mode for workflow validation
  - Add configuration validation commands
  - Create sample data testing functionality
  - _Requirements: All requirements for validation_

- [ ] 11. Implement comprehensive testing suite
- [ ] 11.1 Create unit tests for core components
  - Write tests for VideoData and CuratorState models with validation
  - Create mock YouTube API responses for consistent testing
  - Test quality evaluation algorithms with known datasets
  - Implement transcript processing and analysis tests
  - _Requirements: All requirements need testing coverage_

- [ ] 11.2 Create integration tests for LangGraph workflow
  - Test complete workflow execution with sample data
  - Validate parallel processing and state merging
  - Test iterative refinement loops with various failure scenarios
  - Verify error handling and recovery mechanisms
  - _Requirements: All requirements integrated through workflow testing_

- [ ] 11.3 Add end-to-end testing with live APIs
  - Create rate-limited tests with real YouTube API
  - Test generated podcast script quality assessment
  - Validate container execution and CLI interface testing
  - Add performance benchmarking for processing time
  - _Requirements: 1.1, 2.1, 4.1, 5.1_

- [ ] 12. Create documentation and deployment setup
- [ ] 12.1 Create comprehensive documentation
  - Write README with setup instructions and usage examples
  - Document configuration options and API requirements
  - Create troubleshooting guide for common issues
  - Add examples of generated podcast scripts
  - _Requirements: All requirements need documentation_

- [ ] 12.2 Implement containerization and deployment
  - Create Dockerfile for easy deployment
  - Add docker-compose for development environment
  - Create deployment scripts and configuration templates
  - Implement environment-specific configuration management
  - _Requirements: 5.1 for automated deployment_