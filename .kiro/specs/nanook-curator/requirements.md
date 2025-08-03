# Requirements Document

## Introduction

The nanook-curator is an AI-powered content curation system that automatically discovers trending YouTube videos about AI news, tools, and agents. The system analyzes video content through transcripts, evaluates quality based on engagement metrics and reviews, and generates curated podcast scripts of 5-10 minutes in length. This enables efficient content creation by transforming multiple video sources into cohesive, digestible podcast content.

## Requirements

### Requirement 1

**User Story:** As a content creator, I want the system to automatically discover trending YouTube videos about AI topics, so that I don't have to manually search for relevant content.

#### Acceptance Criteria

1. WHEN the system runs a discovery process THEN it SHALL search for YouTube videos containing keywords related to AI news, tools, and agents
2. WHEN evaluating videos THEN the system SHALL prioritize videos with trending status indicators (view count growth, recent upload date, engagement rate)
3. WHEN filtering results THEN the system SHALL only include videos from the last 30 days to ensure content freshness
4. IF a video has fewer than 1000 views OR is older than 30 days THEN the system SHALL exclude it from consideration

### Requirement 2

**User Story:** As a content curator, I want the system to fetch and analyze video transcripts, so that I can understand the content without watching every video.

#### Acceptance Criteria

1. WHEN a trending video is identified THEN the system SHALL attempt to fetch its transcript using YouTube's API or transcript services
2. IF a video lacks an available transcript THEN the system SHALL skip that video and log the reason
3. WHEN a transcript is obtained THEN the system SHALL parse and clean the text for analysis
4. WHEN processing transcripts THEN the system SHALL extract key topics, main points, and technical details mentioned

### Requirement 3

**User Story:** As a quality-focused curator, I want the system to evaluate content based on interaction metrics and reviews, so that only high-quality content is selected for curation.

#### Acceptance Criteria

1. WHEN evaluating video quality THEN the system SHALL analyze engagement metrics including like-to-dislike ratio, comment sentiment, and view-to-subscriber ratio
2. WHEN assessing content quality THEN the system SHALL score videos based on transcript coherence, technical accuracy indicators, and information density
3. IF a video has a like-to-dislike ratio below 80% THEN the system SHALL lower its quality score
4. WHEN ranking videos THEN the system SHALL combine engagement metrics with content quality scores to create a final ranking

### Requirement 4

**User Story:** As a podcast producer, I want the system to generate a cohesive 5-10 minute podcast script from the best curated content, so that I have ready-to-use material for audio production.

#### Acceptance Criteria

1. WHEN generating a podcast script THEN the system SHALL select the top 3-5 highest-ranked videos as source material
2. WHEN creating the script THEN the system SHALL synthesize information into a coherent narrative flow with introduction, main content, and conclusion
3. WHEN writing the script THEN the system SHALL ensure the final length targets 5-10 minutes of speaking time (approximately 750-1500 words)
4. WHEN structuring content THEN the system SHALL include smooth transitions between topics and cite original video sources
5. IF the generated script exceeds 10 minutes of content THEN the system SHALL automatically trim less important sections while maintaining coherence

### Requirement 5

**User Story:** As a content manager, I want the system to run automatically on a schedule and store results, so that fresh content is always available without manual intervention.

#### Acceptance Criteria

1. WHEN configured THEN the system SHALL run the complete curation process on a configurable schedule (daily, weekly, etc.)
2. WHEN processing is complete THEN the system SHALL store generated podcast scripts with metadata including source videos, generation timestamp, and quality scores
3. WHEN storing results THEN the system SHALL maintain a history of generated scripts and their performance metrics
4. IF the system encounters errors during processing THEN it SHALL log detailed error information and continue with remaining videos
5. WHEN multiple scripts are generated THEN the system SHALL provide a way to compare and select the best script based on quality metrics