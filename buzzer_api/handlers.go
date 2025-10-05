// Additional handlers for AI-driven buzzer detection API
// This file contains the new endpoint handlers for AI analysis features

package main

import (
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

// @Summary Get accounts by buzzer type
// @Description Get all buzzer accounts with specific type (pemerintah, oposisi, komersial, spam, netral)
// @Tags accounts
// @Accept json
// @Produce json
// @Param type path string true "Buzzer type"
// @Param page query int false "Page number" default(1)
// @Param page_size query int false "Page size" default(20)
// @Success 200 {object} SearchResponse
// @Router /api/v1/accounts/type/{type} [get]
func getAccountsByType(c *gin.Context) {
	buzzerType := strings.ToLower(c.Param("type"))
	pageStr := c.DefaultQuery("page", "1")
	pageSizeStr := c.DefaultQuery("page_size", "20")

	page, _ := strconv.Atoi(pageStr)
	pageSize, _ := strconv.Atoi(pageSizeStr)
	offset := (page - 1) * pageSize

	// Validate buzzer type
	validTypes := map[string]bool{
		"pemerintah": true, "oposisi": true, "komersial": true,
		"spam": true, "netral": true,
	}
	if !validTypes[buzzerType] {
		c.JSON(400, gin.H{"error": "Invalid buzzer type. Use: pemerintah, oposisi, komersial, spam, netral"})
		return
	}

	// Count total records
	var total int
	err := db.QueryRow("SELECT COUNT(*) FROM buzzer_accounts WHERE buzzer_type_primary = ?", buzzerType).Scan(&total)
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to count records"})
		return
	}

	// Get paginated results with AI analysis data
	query := `
		SELECT id, username, display_name, social_media_platform, account_url, 
		       posts_count, followers_count, following_count, account_age_days,
		       freq_per_day, burstiness, hashtag_ratio, url_ratio, buzzer_prob,
		       buzzer_prob_enhanced, risk_category, coord_cluster, 
		       coordination_score, network_centrality,
		       buzzer_type_primary, buzzer_type_secondary, activity_pattern, type_confidence,
		       government_score, opposition_score, commercial_score, spam_score,
		       behavioral_anomaly_score, temporal_anomaly_score, engagement_anomaly_score,
		       network_anomaly_score, content_anomaly_score, ai_confidence,
		       vocabulary_diversity, avg_sentence_length, grammar_complexity,
		       content_repetition, emoji_usage, caps_usage,
		       sentiment_positive, sentiment_negative, sentiment_neutral,
		       writing_formal, writing_emotional, writing_persuasive,
		       detected_at, query_used, created_at, updated_at
		FROM buzzer_accounts 
		WHERE buzzer_type_primary = ?
		ORDER BY type_confidence DESC, buzzer_prob_enhanced DESC 
		LIMIT ? OFFSET ?`

	rows, err := db.Query(query, buzzerType, pageSize, offset)
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to query accounts"})
		return
	}
	defer rows.Close()

	var accounts []BuzzerAccount
	for rows.Next() {
		var account BuzzerAccount
		err := scanBuzzerAccount(rows, &account)
		if err != nil {
			continue
		}
		accounts = append(accounts, account)
	}

	totalPages := (total + pageSize - 1) / pageSize

	response := SearchResponse{
		Data:       accounts,
		Total:      total,
		Page:       page,
		PageSize:   pageSize,
		TotalPages: totalPages,
	}

	c.JSON(200, response)
}

// @Summary Get accounts by activity pattern
// @Description Get all buzzer accounts with specific activity pattern (organik, hybrid, artificial)
// @Tags accounts
// @Accept json
// @Produce json
// @Param pattern path string true "Activity pattern"
// @Param page query int false "Page number" default(1)
// @Param page_size query int false "Page size" default(20)
// @Success 200 {object} SearchResponse
// @Router /api/v1/accounts/pattern/{pattern} [get]
func getAccountsByActivityPattern(c *gin.Context) {
	pattern := strings.ToLower(c.Param("pattern"))
	pageStr := c.DefaultQuery("page", "1")
	pageSizeStr := c.DefaultQuery("page_size", "20")

	page, _ := strconv.Atoi(pageStr)
	pageSize, _ := strconv.Atoi(pageSizeStr)
	offset := (page - 1) * pageSize

	// Validate activity pattern
	validPatterns := map[string]bool{"organik": true, "hybrid": true, "artificial": true}
	if !validPatterns[pattern] {
		c.JSON(400, gin.H{"error": "Invalid activity pattern. Use: organik, hybrid, artificial"})
		return
	}

	// Count total records
	var total int
	err := db.QueryRow("SELECT COUNT(*) FROM buzzer_accounts WHERE activity_pattern = ?", pattern).Scan(&total)
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to count records"})
		return
	}

	// Get paginated results
	query := `
		SELECT id, username, display_name, social_media_platform, account_url, 
		       posts_count, followers_count, following_count, account_age_days,
		       freq_per_day, burstiness, hashtag_ratio, url_ratio, buzzer_prob,
		       buzzer_prob_enhanced, risk_category, coord_cluster, 
		       coordination_score, network_centrality,
		       buzzer_type_primary, buzzer_type_secondary, activity_pattern, type_confidence,
		       government_score, opposition_score, commercial_score, spam_score,
		       behavioral_anomaly_score, temporal_anomaly_score, engagement_anomaly_score,
		       network_anomaly_score, content_anomaly_score, ai_confidence,
		       vocabulary_diversity, avg_sentence_length, grammar_complexity,
		       content_repetition, emoji_usage, caps_usage,
		       sentiment_positive, sentiment_negative, sentiment_neutral,
		       writing_formal, writing_emotional, writing_persuasive,
		       detected_at, query_used, created_at, updated_at
		FROM buzzer_accounts 
		WHERE activity_pattern = ?
		ORDER BY behavioral_anomaly_score DESC, buzzer_prob_enhanced DESC 
		LIMIT ? OFFSET ?`

	rows, err := db.Query(query, pattern, pageSize, offset)
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to query accounts"})
		return
	}
	defer rows.Close()

	var accounts []BuzzerAccount
	for rows.Next() {
		var account BuzzerAccount
		err := scanBuzzerAccount(rows, &account)
		if err != nil {
			continue
		}
		accounts = append(accounts, account)
	}

	totalPages := (total + pageSize - 1) / pageSize

	response := SearchResponse{
		Data:       accounts,
		Total:      total,
		Page:       page,
		PageSize:   pageSize,
		TotalPages: totalPages,
	}

	c.JSON(200, response)
}

// @Summary Get post analysis
// @Description Get detailed AI analysis for a specific post
// @Tags posts
// @Accept json
// @Produce json
// @Param postId path string true "Post ID"
// @Success 200 {object} PostAnalysis
// @Router /api/v1/posts/analysis/{postId} [get]
func getPostAnalysis(c *gin.Context) {
	postID := c.Param("postId")

	query := `
		SELECT id, post_id, username, social_media_platform, content_hash,
		       sentiment_score, emotion_score, sophistication_score, spam_indicators,
		       promotional_score, analysis_data, query_used, created_at
		FROM post_analysis 
		WHERE post_id = ?`

	var analysis PostAnalysis
	err := db.QueryRow(query, postID).Scan(
		&analysis.ID, &analysis.PostID, &analysis.Username, &analysis.Platform,
		&analysis.ContentHash, &analysis.SentimentScore, &analysis.EmotionScore,
		&analysis.SophisticationScore, &analysis.SpamIndicators, &analysis.PromotionalScore,
		&analysis.AnalysisData, &analysis.QueryUsed, &analysis.CreatedAt,
	)

	if err != nil {
		c.JSON(404, gin.H{"error": "Post analysis not found"})
		return
	}

	c.JSON(200, analysis)
}

// @Summary Get AI analysis details for user
// @Description Get comprehensive AI analysis details for a specific user
// @Tags ai-analysis
// @Accept json
// @Produce json
// @Param username path string true "Username"
// @Success 200 {object} AIAnalysisDetails
// @Router /api/v1/ai/analysis/{username} [get]
func getAIAnalysisDetails(c *gin.Context) {
	username := c.Param("username")

	query := `
		SELECT id, username, social_media_platform, analysis_type, analysis_data,
		       confidence_score, query_used, created_at
		FROM ai_analysis_details 
		WHERE username = ?
		ORDER BY created_at DESC`

	rows, err := db.Query(query, username)
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to query AI analysis"})
		return
	}
	defer rows.Close()

	var analyses []AIAnalysisDetails
	for rows.Next() {
		var analysis AIAnalysisDetails
		err := rows.Scan(
			&analysis.ID, &analysis.Username, &analysis.Platform,
			&analysis.AnalysisType, &analysis.AnalysisData, &analysis.ConfidenceScore,
			&analysis.QueryUsed, &analysis.CreatedAt,
		)
		if err != nil {
			continue
		}
		analyses = append(analyses, analysis)
	}

	if len(analyses) == 0 {
		c.JSON(404, gin.H{"error": "No AI analysis found for user"})
		return
	}

	c.JSON(200, analyses)
}

// @Summary Get content analysis statistics
// @Description Get overall content analysis statistics
// @Tags ai-analysis
// @Accept json
// @Produce json
// @Success 200 {object} map[string]interface{}
// @Router /api/v1/ai/content-analysis [get]
func getContentAnalysisStats(c *gin.Context) {
	query := `
		SELECT 
			AVG(vocabulary_diversity) as avg_vocab_diversity,
			AVG(avg_sentence_length) as avg_sentence_len,
			AVG(grammar_complexity) as avg_grammar_complexity,
			AVG(content_repetition) as avg_content_repetition,
			AVG(emoji_usage) as avg_emoji_usage,
			AVG(caps_usage) as avg_caps_usage,
			COUNT(*) as total_accounts
		FROM buzzer_accounts`

	var stats map[string]interface{}
	var avgVocab, avgSentence, avgGrammar, avgRepetition, avgEmoji, avgCaps float64
	var total int

	err := db.QueryRow(query).Scan(
		&avgVocab, &avgSentence, &avgGrammar, &avgRepetition, &avgEmoji, &avgCaps, &total,
	)

	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to query content analysis stats"})
		return
	}

	stats = map[string]interface{}{
		"average_vocabulary_diversity": avgVocab,
		"average_sentence_length":      avgSentence,
		"average_grammar_complexity":   avgGrammar,
		"average_content_repetition":   avgRepetition,
		"average_emoji_usage":          avgEmoji,
		"average_caps_usage":           avgCaps,
		"total_accounts":               total,
	}

	c.JSON(200, stats)
}

// @Summary Get sentiment analysis statistics
// @Description Get overall sentiment analysis statistics
// @Tags ai-analysis
// @Accept json
// @Produce json
// @Success 200 {object} map[string]interface{}
// @Router /api/v1/ai/sentiment-analysis [get]
func getSentimentAnalysisStats(c *gin.Context) {
	query := `
		SELECT 
			AVG(sentiment_positive) as avg_positive,
			AVG(sentiment_negative) as avg_negative,
			AVG(sentiment_neutral) as avg_neutral,
			buzzer_type_primary,
			COUNT(*) as count
		FROM buzzer_accounts 
		GROUP BY buzzer_type_primary`

	rows, err := db.Query(query)
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to query sentiment analysis stats"})
		return
	}
	defer rows.Close()

	var stats []map[string]interface{}
	for rows.Next() {
		var avgPos, avgNeg, avgNeut float64
		var buzzerType string
		var count int

		err := rows.Scan(&avgPos, &avgNeg, &avgNeut, &buzzerType, &count)
		if err != nil {
			continue
		}

		stats = append(stats, map[string]interface{}{
			"buzzer_type":      buzzerType,
			"average_positive": avgPos,
			"average_negative": avgNeg,
			"average_neutral":  avgNeut,
			"account_count":    count,
		})
	}

	c.JSON(200, stats)
}

// @Summary Get behavioral analysis statistics
// @Description Get behavioral anomaly analysis statistics
// @Tags ai-analysis
// @Accept json
// @Produce json
// @Success 200 {object} map[string]interface{}
// @Router /api/v1/ai/behavioral-analysis [get]
func getBehavioralAnalysisStats(c *gin.Context) {
	query := `
		SELECT 
			AVG(behavioral_anomaly_score) as avg_behavioral,
			AVG(temporal_anomaly_score) as avg_temporal,
			AVG(engagement_anomaly_score) as avg_engagement,
			AVG(network_anomaly_score) as avg_network,
			AVG(content_anomaly_score) as avg_content,
			activity_pattern,
			COUNT(*) as count
		FROM buzzer_accounts 
		GROUP BY activity_pattern`

	rows, err := db.Query(query)
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to query behavioral analysis stats"})
		return
	}
	defer rows.Close()

	var stats []map[string]interface{}
	for rows.Next() {
		var avgBehavioral, avgTemporal, avgEngagement, avgNetwork, avgContent float64
		var pattern string
		var count int

		err := rows.Scan(&avgBehavioral, &avgTemporal, &avgEngagement, &avgNetwork, &avgContent, &pattern, &count)
		if err != nil {
			continue
		}

		stats = append(stats, map[string]interface{}{
			"activity_pattern":         pattern,
			"average_behavioral_score": avgBehavioral,
			"average_temporal_score":   avgTemporal,
			"average_engagement_score": avgEngagement,
			"average_network_score":    avgNetwork,
			"average_content_score":    avgContent,
			"account_count":            count,
		})
	}

	c.JSON(200, stats)
}

// @Summary Get buzzer type statistics
// @Description Get detailed statistics by buzzer type
// @Tags statistics
// @Accept json
// @Produce json
// @Success 200 {array} BuzzerTypeStats
// @Router /api/v1/stats/types [get]
func getBuzzerTypeStats(c *gin.Context) {
	query := `
		SELECT buzzer_type_primary,
		       COUNT(*) as count,
		       AVG(buzzer_prob_enhanced) as avg_prob,
		       AVG(type_confidence) as avg_confidence
		FROM buzzer_accounts 
		GROUP BY buzzer_type_primary
		ORDER BY count DESC`

	rows, err := db.Query(query)
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to query buzzer type statistics"})
		return
	}
	defer rows.Close()

	var stats []BuzzerTypeStats
	for rows.Next() {
		var stat BuzzerTypeStats
		err := rows.Scan(&stat.BuzzerType, &stat.Count, &stat.AverageProb, &stat.AverageConfidence)
		if err != nil {
			continue
		}
		stats = append(stats, stat)
	}

	c.JSON(200, stats)
}

// @Summary Get activity pattern statistics
// @Description Get detailed statistics by activity pattern
// @Tags statistics
// @Accept json
// @Produce json
// @Success 200 {array} ActivityPatternStats
// @Router /api/v1/stats/patterns [get]
func getActivityPatternStats(c *gin.Context) {
	query := `
		SELECT activity_pattern,
		       COUNT(*) as count,
		       AVG(behavioral_anomaly_score) as avg_score
		FROM buzzer_accounts 
		GROUP BY activity_pattern
		ORDER BY count DESC`

	rows, err := db.Query(query)
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to query activity pattern statistics"})
		return
	}
	defer rows.Close()

	var stats []ActivityPatternStats
	for rows.Next() {
		var stat ActivityPatternStats
		err := rows.Scan(&stat.Pattern, &stat.Count, &stat.AverageScore)
		if err != nil {
			continue
		}
		stats = append(stats, stat)
	}

	c.JSON(200, stats)
}

// @Summary Get AI metrics statistics
// @Description Get comprehensive AI analysis metrics
// @Tags statistics
// @Accept json
// @Produce json
// @Success 200 {object} map[string]interface{}
// @Router /api/v1/stats/ai-metrics [get]
func getAIMetricsStats(c *gin.Context) {
	query := `
		SELECT 
			AVG(ai_confidence) as avg_ai_confidence,
			AVG(type_confidence) as avg_type_confidence,
			COUNT(CASE WHEN ai_confidence > 0.8 THEN 1 END) as high_confidence_count,
			COUNT(CASE WHEN ai_confidence > 0.5 THEN 1 END) as medium_confidence_count,
			COUNT(*) as total_accounts,
			MAX(updated_at) as last_analysis
		FROM buzzer_accounts`

	var avgAIConf, avgTypeConf float64
	var highConf, mediumConf, total int
	var lastAnalysis time.Time

	err := db.QueryRow(query).Scan(
		&avgAIConf, &avgTypeConf, &highConf, &mediumConf, &total, &lastAnalysis,
	)

	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to query AI metrics stats"})
		return
	}

	stats := map[string]interface{}{
		"average_ai_confidence":      avgAIConf,
		"average_type_confidence":    avgTypeConf,
		"high_confidence_accounts":   highConf,
		"medium_confidence_accounts": mediumConf,
		"total_accounts":             total,
		"confidence_distribution": map[string]float64{
			"high":   float64(highConf) / float64(total) * 100,
			"medium": float64(mediumConf) / float64(total) * 100,
			"low":    float64(total-mediumConf) / float64(total) * 100,
		},
		"last_analysis": lastAnalysis,
	}

	c.JSON(200, stats)
}

// @Summary Advanced search with AI filters
// @Description Advanced search with AI analysis filters
// @Tags search
// @Accept json
// @Produce json
// @Param q query string false "Search query"
// @Param buzzer_type query string false "Buzzer type filter"
// @Param activity_pattern query string false "Activity pattern filter"
// @Param min_ai_confidence query float64 false "Minimum AI confidence"
// @Param sentiment query string false "Dominant sentiment (positive, negative, neutral)"
// @Param page query int false "Page number" default(1)
// @Param page_size query int false "Page size" default(20)
// @Success 200 {object} SearchResponse
// @Router /api/v1/search/advanced [get]
func advancedSearch(c *gin.Context) {
	query := c.Query("q")
	buzzerType := c.Query("buzzer_type")
	activityPattern := c.Query("activity_pattern")
	minAIConfStr := c.Query("min_ai_confidence")
	sentiment := c.Query("sentiment")
	pageStr := c.DefaultQuery("page", "1")
	pageSizeStr := c.DefaultQuery("page_size", "20")

	page, _ := strconv.Atoi(pageStr)
	pageSize, _ := strconv.Atoi(pageSizeStr)
	offset := (page - 1) * pageSize

	var minAIConf float64
	if minAIConfStr != "" {
		minAIConf, _ = strconv.ParseFloat(minAIConfStr, 64)
	}

	// Build advanced WHERE clause
	var conditions []string
	var args []interface{}

	if query != "" {
		conditions = append(conditions, "(username LIKE ? OR display_name LIKE ?)")
		args = append(args, "%"+query+"%", "%"+query+"%")
	}
	if buzzerType != "" {
		conditions = append(conditions, "buzzer_type_primary = ?")
		args = append(args, buzzerType)
	}
	if activityPattern != "" {
		conditions = append(conditions, "activity_pattern = ?")
		args = append(args, activityPattern)
	}
	if minAIConf > 0 {
		conditions = append(conditions, "ai_confidence >= ?")
		args = append(args, minAIConf)
	}
	if sentiment != "" {
		switch sentiment {
		case "positive":
			conditions = append(conditions, "sentiment_positive > sentiment_negative AND sentiment_positive > sentiment_neutral")
		case "negative":
			conditions = append(conditions, "sentiment_negative > sentiment_positive AND sentiment_negative > sentiment_neutral")
		case "neutral":
			conditions = append(conditions, "sentiment_neutral > sentiment_positive AND sentiment_neutral > sentiment_negative")
		}
	}

	whereClause := ""
	if len(conditions) > 0 {
		whereClause = "WHERE " + strings.Join(conditions, " AND ")
	}

	// Count total records
	countQuery := `SELECT COUNT(*) FROM buzzer_accounts ` + whereClause
	var total int
	err := db.QueryRow(countQuery, args...).Scan(&total)
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to count records"})
		return
	}

	// Get paginated results
	sqlQuery := `
		SELECT id, username, display_name, social_media_platform, account_url, 
		       posts_count, followers_count, following_count, account_age_days,
		       freq_per_day, burstiness, hashtag_ratio, url_ratio, buzzer_prob,
		       buzzer_prob_enhanced, risk_category, coord_cluster, 
		       coordination_score, network_centrality,
		       buzzer_type_primary, buzzer_type_secondary, activity_pattern, type_confidence,
		       government_score, opposition_score, commercial_score, spam_score,
		       behavioral_anomaly_score, temporal_anomaly_score, engagement_anomaly_score,
		       network_anomaly_score, content_anomaly_score, ai_confidence,
		       vocabulary_diversity, avg_sentence_length, grammar_complexity,
		       content_repetition, emoji_usage, caps_usage,
		       sentiment_positive, sentiment_negative, sentiment_neutral,
		       writing_formal, writing_emotional, writing_persuasive,
		       detected_at, query_used, created_at, updated_at
		FROM buzzer_accounts ` + whereClause + `
		ORDER BY ai_confidence DESC, buzzer_prob_enhanced DESC 
		LIMIT ? OFFSET ?`

	args = append(args, pageSize, offset)
	rows, err := db.Query(sqlQuery, args...)
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to query accounts"})
		return
	}
	defer rows.Close()

	var accounts []BuzzerAccount
	for rows.Next() {
		var account BuzzerAccount
		err := scanBuzzerAccount(rows, &account)
		if err != nil {
			continue
		}
		accounts = append(accounts, account)
	}

	totalPages := (total + pageSize - 1) / pageSize

	response := SearchResponse{
		Data:       accounts,
		Total:      total,
		Page:       page,
		PageSize:   pageSize,
		TotalPages: totalPages,
	}

	c.JSON(200, response)
}

// Helper function to scan BuzzerAccount from database rows
func scanBuzzerAccount(rows interface{}, account *BuzzerAccount) error {
	// This is a type assertion - in real implementation you'd use sql.Rows.Scan
	// For now, we'll implement a simplified version
	if scanner, ok := rows.(interface {
		Scan(dest ...interface{}) error
	}); ok {
		return scanner.Scan(
			&account.ID, &account.Username, &account.DisplayName,
			&account.SocialMediaPlatform, &account.AccountURL, &account.PostsCount,
			&account.FollowersCount, &account.FollowingCount, &account.AccountAgeDays,
			&account.FreqPerDay, &account.Burstiness, &account.HashtagRatio,
			&account.URLRatio, &account.BuzzerProb, &account.BuzzerProbEnhanced,
			&account.RiskCategory, &account.CoordCluster, &account.CoordinationScore,
			&account.NetworkCentrality,
			&account.BuzzerTypePrimary, &account.BuzzerTypeSecondary, &account.ActivityPattern, &account.TypeConfidence,
			&account.GovernmentScore, &account.OppositionScore, &account.CommercialScore, &account.SpamScore,
			&account.BehavioralAnomalyScore, &account.TemporalAnomalyScore, &account.EngagementAnomalyScore,
			&account.NetworkAnomalyScore, &account.ContentAnomalyScore, &account.AIConfidence,
			&account.VocabularyDiversity, &account.AvgSentenceLength, &account.GrammarComplexity,
			&account.ContentRepetition, &account.EmojiUsage, &account.CapsUsage,
			&account.SentimentPositive, &account.SentimentNegative, &account.SentimentNeutral,
			&account.WritingFormal, &account.WritingEmotional, &account.WritingPersuasive,
			&account.DetectedAt, &account.QueryUsed, &account.CreatedAt, &account.UpdatedAt,
		)
	}
	return nil
}

// Update the enhanced stats function
func getEnhancedStats(c *gin.Context) {
	stats := StatsResponse{
		RiskDistribution:  make(map[string]int),
		PlatformStats:     make(map[string]int),
		AIAnalysisEnabled: true,
	}

	// Total accounts
	err := db.QueryRow("SELECT COUNT(*) FROM buzzer_accounts").Scan(&stats.TotalAccounts)
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to get total accounts"})
		return
	}

	// Total posts
	err = db.QueryRow("SELECT COUNT(*) FROM posts").Scan(&stats.TotalPosts)
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to get total posts"})
		return
	}

	// Risk distribution
	rows, err := db.Query("SELECT risk_category, COUNT(*) FROM buzzer_accounts GROUP BY risk_category")
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to get risk distribution"})
		return
	}
	defer rows.Close()

	for rows.Next() {
		var riskLevel string
		var count int
		rows.Scan(&riskLevel, &count)
		stats.RiskDistribution[riskLevel] = count
	}

	// Platform statistics
	rows, err = db.Query("SELECT social_media_platform, COUNT(*) FROM buzzer_accounts GROUP BY social_media_platform")
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to get platform stats"})
		return
	}
	defer rows.Close()

	for rows.Next() {
		var platform string
		var count int
		rows.Scan(&platform, &count)
		stats.PlatformStats[platform] = count
	}

	// Buzzer type statistics
	typeRows, err := db.Query(`
		SELECT buzzer_type_primary, COUNT(*) as count, 
		       AVG(buzzer_prob_enhanced) as avg_prob, AVG(type_confidence) as avg_conf
		FROM buzzer_accounts GROUP BY buzzer_type_primary`)
	if err == nil {
		defer typeRows.Close()
		for typeRows.Next() {
			var stat BuzzerTypeStats
			typeRows.Scan(&stat.BuzzerType, &stat.Count, &stat.AverageProb, &stat.AverageConfidence)
			stats.BuzzerTypeStats = append(stats.BuzzerTypeStats, stat)
		}
	}

	// Activity pattern statistics
	patternRows, err := db.Query(`
		SELECT activity_pattern, COUNT(*) as count, AVG(behavioral_anomaly_score) as avg_score
		FROM buzzer_accounts GROUP BY activity_pattern`)
	if err == nil {
		defer patternRows.Close()
		for patternRows.Next() {
			var stat ActivityPatternStats
			patternRows.Scan(&stat.Pattern, &stat.Count, &stat.AverageScore)
			stats.ActivityPatternStats = append(stats.ActivityPatternStats, stat)
		}
	}

	// Coordination groups
	err = db.QueryRow("SELECT COUNT(DISTINCT group_id) FROM coordination_groups").Scan(&stats.CoordinationGroups)
	if err != nil {
		stats.CoordinationGroups = 0
	}

	// Last update
	err = db.QueryRow("SELECT MAX(updated_at) FROM buzzer_accounts").Scan(&stats.LastUpdate)
	if err != nil {
		stats.LastUpdate = time.Now()
	}

	c.JSON(200, stats)
}
